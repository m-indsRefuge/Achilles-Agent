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

            // 3. Selective Reinforcement: Record 'seen' results for this suggestion
            // In a real scenario, we would wait for the user to 'accept' (Tab) to send positive signal.
            // For this implementation, we log the retrieval.
            if (kbResults && kbResults.length > 0) {
                const retrievedIds = kbResults.map((r: any) => r.id);
                this.bridge.queryMemory('feedback', {
                    query: lineText.trim(),
                    retrieved_ids: retrievedIds,
                    selected_ids: [], // Placeholder for user interaction
                    dismissed_ids: []
                }).catch(e => console.error("Inline feedback error:", e));
            }

            // 4. Return as inline item
            return [new vscode.InlineCompletionItem(completion, new vscode.Range(position, position))];
        } catch (error) {
            console.error('Inline completion error:', error);
            return [];
        }
    }
}
