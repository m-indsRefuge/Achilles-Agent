import * as vscode from 'vscode';
import { ModelInterface } from './ai/codeLlama/modelInterface';
import { PythonBridge } from './comms/pythonBridge';
import { fillTemplate, CHAT_PROMPT } from './ai/prompts/promptTemplates';
import { SidebarProvider } from './ui/sidebarProvider';
import { ProjectAnalyzer } from './project/analyzer';

export function activate(context: vscode.ExtensionContext) {
    console.log('Achilles Agent is now active!');

    const model = new ModelInterface({
        modelName: 'code-llama-7b',
        temperature: 0.7,
        maxTokens: 512
    });

    const bridge = new PythonBridge();
    const analyzer = new ProjectAnalyzer();

    const sidebarProvider = new SidebarProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(SidebarProvider.viewType, sidebarProvider)
    );

    let askCommand = vscode.commands.registerCommand('achilles.ask', async (text?: string) => {
        const input = text || await vscode.window.showInputBox({
            prompt: 'Ask Achilles anything...',
            placeHolder: 'e.g. How do I optimize this loop?'
        });

        if (input) {
            try {
                sidebarProvider.addMessage('user', input);

                // 1. Search Knowledge Base via Python Bridge
                const kbResults = await bridge.queryMemory('kb', { text: input });
                const contextText = kbResults.length > 0
                    ? kbResults.map((r: any) => r.text).join('\n')
                    : 'No relevant context found.';

                // 2. Prepare Prompt
                const prompt = fillTemplate(CHAT_PROMPT, {
                    context: contextText,
                    question: input
                });

                // 3. Generate Response
                const response = await model.generate(prompt);

                sidebarProvider.addMessage('bot', response);
            } catch (error: any) {
                vscode.window.showErrorMessage(`Achilles Error: ${error.message}`);
            }
        }
    });

    let clearMemoryCommand = vscode.commands.registerCommand('achilles.clearMemory', () => {
        vscode.window.showInformationMessage('Achilles short-term memory cleared.');
    });

    let analyzeCommand = vscode.commands.registerCommand('achilles.analyzeProject', async () => {
        vscode.window.showInformationMessage('Achilles is analyzing your project...');
        try {
            await analyzer.analyzeWorkspace();
            vscode.window.showInformationMessage('Project analysis complete!');
        } catch (error: any) {
            vscode.window.showErrorMessage(`Analysis Error: ${error.message}`);
        }
    });

    context.subscriptions.push(askCommand, clearMemoryCommand, analyzeCommand);

    // Initial analysis
    analyzer.analyzeWorkspace().catch(e => console.error('Initial analysis failed:', e));
}

export function deactivate() {}
