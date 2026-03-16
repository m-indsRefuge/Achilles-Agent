import { ChatGPTClient } from '../../comms/chatGPTClient';

export interface ModelConfig {
    modelName: string;
    temperature: number;
    maxTokens: number;
}

import * as vscode from 'vscode';

export class ModelInterface {
    private config: ModelConfig;
    private client: ChatGPTClient;

    constructor(config: ModelConfig) {
        this.config = config;
        const extensionConfig = vscode.workspace.getConfiguration('achilles');
        const baseURL = extensionConfig.get<string>('ollama.baseURL');
        this.client = new ChatGPTClient('ollama', baseURL);
    }

    async generate(prompt: string): Promise<string> {
        return this.client.chat(
            [{ role: 'user', content: prompt }],
            this.config.modelName,
            this.config.temperature
        );
    }

    async embed(text: string): Promise<number[]> {
        return this.client.createEmbedding(text);
    }
}
