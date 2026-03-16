import * as vscode from 'vscode';
import { ModelInterface } from './ai/codeLlama/modelInterface';
import { PythonBridge } from './comms/pythonBridge';
import { fillTemplate, CHAT_PROMPT } from './ai/prompts/promptTemplates';
import { SidebarProvider } from './ui/sidebarProvider';
import { ProjectAnalyzer } from './project/analyzer';
import { FileWatcher } from './project/fileWatcher';
import { TaskRunner } from './tasks/taskRunner';
import { ReRanker } from './ai/reRanker/reRankerInterface';
import { AchillesMCPServer } from './comms/mcpServer';

export function activate(context: vscode.ExtensionContext) {
    console.log('Achilles Agent is now active!');

    const model = new ModelInterface({
        modelName: 'code-llama-7b',
        temperature: 0.7,
        maxTokens: 512
    });

    const bridge = new PythonBridge();
    const analyzer = new ProjectAnalyzer(bridge);
    const fileWatcher = new FileWatcher(analyzer);
    const mcpServer = new AchillesMCPServer(bridge);
    const taskRunner = new TaskRunner();
    const reranker = new ReRanker();

    const sidebarProvider = new SidebarProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(SidebarProvider.viewType, sidebarProvider),
        bridge,
        fileWatcher
    );

    let askCommand = vscode.commands.registerCommand('achilles.ask', async (text?: string) => {
        const input = text || await vscode.window.showInputBox({
            prompt: 'Ask Achilles anything...',
            placeHolder: 'e.g. How do I optimize this loop?'
        });

        if (input) {
            try {
                sidebarProvider.addMessage('user', input);

                let currentInput = input;
                let loopCount = 0;
                const MAX_LOOPS = 3;

                while (loopCount < MAX_LOOPS) {
                    // 1. Search Knowledge Base
                    const kbResults = await bridge.queryMemory('kb', { text: currentInput, top_k: 10 });
                    const refinedResults = await reranker.rerank(currentInput, kbResults);
                    const topResults = refinedResults.slice(0, 5);
                    const contextText = topResults.length > 0 ? topResults.map((r: any) => r.text).join('\n') : 'No relevant context found.';

                    // 2. Generate Response
                    const prompt = fillTemplate(CHAT_PROMPT, { context: contextText, question: currentInput });
                    const response = await model.generate(prompt);

                    // 3. Handle Tasks
                    if (response.includes('RUN_TASK:')) {
                        const taskJson = response.split('RUN_TASK:')[1].trim();
                        try {
                            const task = JSON.parse(taskJson);
                            const taskResult = await taskRunner.run(task);
                            sidebarProvider.addMessage('bot', `${response}\n\nTask Result: ${taskResult}`);

                            // Feed the result back to the AI for the next step
                            currentInput = `Task result: ${taskResult}. What is the next step?`;
                            loopCount++;
                        } catch (e: any) {
                            sidebarProvider.addMessage('bot', `${response}\n\nTask Failed: ${e.message}`);
                            break;
                        }
                    } else {
                        sidebarProvider.addMessage('bot', response);
                        break;
                    }
                }
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

    // Initial actions
    analyzer.analyzeWorkspace().catch(e => console.error('Initial analysis failed:', e));
    mcpServer.run().catch(e => console.error('MCP Server failed to start:', e));
}

export function deactivate() {}
