import { ChatGPTClient } from '../../comms/chatGPTClient';

export interface ModelConfig {
    modelName: string;
    temperature: number;
    maxTokens: number;
}

export class ModelInterface {
    private config: ModelConfig;
    private client: ChatGPTClient;

    constructor(config: ModelConfig) {
        this.config = config;
        this.client = new ChatGPTClient();
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
