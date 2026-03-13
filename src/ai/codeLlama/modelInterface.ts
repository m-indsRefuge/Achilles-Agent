export interface ModelConfig {
    modelName: string;
    temperature: number;
    maxTokens: number;
}

export class ModelInterface {
    private config: ModelConfig;

    constructor(config: ModelConfig) {
        this.config = config;
    }

    async generate(prompt: string): Promise<string> {
        // Placeholder for actual LLM call
        console.log(`Generating with ${this.config.modelName} for prompt: ${prompt}`);
        return `Response from ${this.config.modelName} to: ${prompt}`;
    }

    async embed(text: string): Promise<number[]> {
        // Placeholder for actual embedding call
        return new Array(384).fill(0);
    }
}
