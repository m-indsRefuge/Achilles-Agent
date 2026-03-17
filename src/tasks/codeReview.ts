import * as vscode from 'vscode';
import { ModelInterface } from '../ai/codeLlama/modelInterface';
import { PythonBridge } from '../comms/pythonBridge';
import { ReRanker } from '../ai/reRanker/reRankerInterface';

export class CodeReviewer {
    private model: ModelInterface;
    private bridge: PythonBridge;
    private reranker: ReRanker;

    constructor(model: ModelInterface, bridge: PythonBridge, reranker: ReRanker) {
        this.model = model;
        this.bridge = bridge;
        this.reranker = reranker;
    }

    public async reviewFile(document: vscode.TextDocument): Promise<string> {
        const fileContent = document.getText();
        const fileName = document.fileName;

        // 1. Find similar patterns in Knowledge Base for context
        const kbResults = await this.bridge.queryMemory('kb', { text: fileContent.slice(0, 500), top_k: 10 });
        const refinedResults = await this.reranker.rerank(fileContent.slice(0, 500), kbResults);
        const contextText = refinedResults.slice(0, 3).map((r: any) => r.text).join('\n');

        // 2. Request AI Review
        const prompt = `You are an expert code reviewer. Review the following file: ${fileName}

        Use the following project context for style and consistency reference:
        ${contextText}

        File Content:
        ${fileContent}

        Provide a list of 3-5 specific improvements or bugs.`;

        const review = await this.model.generate(prompt);
        return review;
    }
}
