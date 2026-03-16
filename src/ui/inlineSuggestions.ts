import * as vscode from 'vscode';
import { ModelInterface } from '../ai/codeLlama/modelInterface';
import { PythonBridge } from '../comms/pythonBridge';

export class InlineCompletionProvider implements vscode.InlineCompletionItemProvider {
    private model: ModelInterface;
    private bridge: PythonBridge;

    constructor(model: ModelInterface, bridge: PythonBridge) {
        this.model = model;
        this.bridge = bridge;
    }

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[]> {

        // Only provide suggestions if user typed at least 10 characters in the line
        const lineText = document.lineAt(position.line).text;
        if (position.character < 10) {
            return [];
        }

        try {
            // 1. Get project context from Memory
            const kbResults = await this.bridge.queryMemory('kb', {
                text: lineText.trim(),
                top_k: 2
            });
            const contextText = kbResults.map((r: any) => r.text).join('\n');

            // 2. Request completion from LLM
            const prompt = `Context: ${contextText}\n\nCode so far: ${lineText}\nNext code:`;
            const completion = await this.model.generate(prompt);

            // 3. Return as inline item
            return [new vscode.InlineCompletionItem(completion, new vscode.Range(position, position))];
        } catch (error) {
            console.error('Inline completion error:', error);
            return [];
        }
    }
}
