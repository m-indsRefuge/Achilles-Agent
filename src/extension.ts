import * as vscode from 'vscode';
import { ModelInterface } from './ai/codeLlama/modelInterface';
import { PythonBridge } from './comms/pythonBridge';
import { fillTemplate, CHAT_PROMPT } from './ai/prompts/promptTemplates';
import { SidebarProvider } from './ui/sidebarProvider';
import { ProjectAnalyzer } from './project/analyzer';
import { FileWatcher } from './project/fileWatcher';
import { TaskRunner } from './tasks/taskRunner';
import { AutomationManager } from './tasks/automation';
import { CodeReviewer } from './tasks/codeReview';
import { ReRanker } from './ai/reRanker/reRankerInterface';
import { AchillesMCPServer } from './comms/mcpServer';
import { InlineCompletionProvider } from './ui/inlineSuggestions';

export function activate(context: vscode.ExtensionContext) {
    console.log('Achilles Agent is now active!');

    const config = vscode.workspace.getConfiguration('achilles');

    const model = new ModelInterface({
        modelName: config.get<string>('ollama.model') || 'codellama',
        temperature: 0.7,
        maxTokens: 512
    });

    const bridge = new PythonBridge();
    const analyzer = new ProjectAnalyzer(bridge);
    const fileWatcher = new FileWatcher(analyzer);
    const mcpServer = new AchillesMCPServer(bridge);
    const taskRunner = new TaskRunner();
    const automationManager = new AutomationManager(taskRunner, model);
    const reranker = new ReRanker(bridge);
    const codeReviewer = new CodeReviewer(model, bridge, reranker);

    const sidebarProvider = new SidebarProvider(context.extensionUri);
    const inlineProvider = new InlineCompletionProvider(model, bridge);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(SidebarProvider.viewType, sidebarProvider),
        vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, inlineProvider),
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
                // Get Editor Context
                const editor = vscode.window.activeTextEditor;
                let contextInput = input;
                if (editor) {
                    const selection = editor.document.getText(editor.selection);
                    contextInput = `[File: ${editor.document.fileName}, Language: ${editor.document.languageId}]\n` +
                                 (selection ? `[Selected Code: ${selection}]\n` : "") +
                                 input;
                }

                sidebarProvider.addMessage('user', input);

                // Add to STM
                await bridge.queryMemory('stm_add', { entry: { role: 'user', content: contextInput } });

                let currentInput = contextInput;
                let loopCount = 0;
                const MAX_LOOPS = 3;

                const topK = config.get<number>('search.topK') || 5;

                while (loopCount < MAX_LOOPS) {
                    // 1. Search Knowledge Base
                    const kbResults = await bridge.queryMemory('kb', { text: currentInput, top_k: topK * 2 });
                    const refinedResults = await reranker.rerank(currentInput, kbResults);
                    const topResults = refinedResults.slice(0, topK);
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
                            await bridge.queryMemory('stm_add', { entry: { role: 'system', content: currentInput } });
                            loopCount++;
                        } catch (e: any) {
                            sidebarProvider.addMessage('bot', `${response}\n\nTask Failed: ${e.message}`);
                            break;
                        }
                    } else {
                        sidebarProvider.addMessage('bot', response);
                        await bridge.queryMemory('stm_add', { entry: { role: 'assistant', content: response } });
                        break;
                    }
                }

                // Check for summarization (every 10 messages for simplicity in skeleton)
                const stm = await bridge.queryMemory('stm', { key: 'role', value: 'user' });
                if (stm.length > 10) {
                    const summary = await model.generate('Summarize our conversation so far in 1-2 sentences.');
                    await bridge.queryMemory('stm_summarize', { text: summary });
                }
            } catch (error: any) {
                vscode.window.showErrorMessage(`Achilles Error: ${error.message}`);
            }
        }
    });

    let clearMemoryCommand = vscode.commands.registerCommand('achilles.clearMemory', async () => {
        try {
            await bridge.queryMemory('clear_all', {});
            vscode.window.showInformationMessage('Achilles memory cleared.');
        } catch (error: any) {
            vscode.window.showErrorMessage(`Clear Memory Error: ${error.message}`);
        }
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

    let workflowCommand = vscode.commands.registerCommand('achilles.runWorkflow', async () => {
        const goal = await vscode.window.showInputBox({ prompt: 'Enter workflow goal (e.g. Refactor API)' });
        if (goal) {
            sidebarProvider.addMessage('user', `Goal: ${goal}`);
            const results = await automationManager.runWorkflow(goal);
            results.forEach(res => sidebarProvider.addMessage('bot', res));
        }
    });

    let trainCommand = vscode.commands.registerCommand('achilles.trainOnProject', async () => {
        const modelName = config.get<string>('ollama.model') || 'codellama';
        vscode.window.showInformationMessage(`Achilles is training on your project using ${modelName}...`);
        try {
            await bridge.queryMemory('train_on_kb', { model: modelName });
            vscode.window.showInformationMessage('Fine-tuning complete!');
        } catch (error: any) {
            vscode.window.showErrorMessage(`Training Error: ${error.message}`);
        }
    });

    let reviewCommand = vscode.commands.registerCommand('achilles.reviewFile', async () => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            vscode.window.showInformationMessage(`Achilles is reviewing ${editor.document.fileName}...`);
            const review = await codeReviewer.reviewFile(editor.document);
            sidebarProvider.addMessage('bot', `### Code Review: ${editor.document.fileName}\n\n${review}`);
        }
    });

    context.subscriptions.push(askCommand, clearMemoryCommand, analyzeCommand, workflowCommand, trainCommand, reviewCommand);

    // Initial actions
    analyzer.analyzeWorkspace().catch(e => console.error('Initial analysis failed:', e));
    // In VS Code, we avoid running Stdio MCP server directly in the extension process.
    // mcpServer.run().catch(e => console.error('MCP Server failed to start:', e));
}

export function deactivate() {}
