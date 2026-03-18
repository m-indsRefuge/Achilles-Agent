import axios from 'axios';

export class ChatGPTClient {
    private apiKey: string;
    private baseURL: string;

    constructor(apiKey: string = 'ollama', baseURL: string = 'http://localhost:11434/v1') {
        this.apiKey = apiKey;
        this.baseURL = baseURL;
    }

    setBaseURL(url: string) {
        this.baseURL = url;
    }

    async chat(messages: any[], model: string = 'codellama', temperature: number = 0.7): Promise<string> {
        try {
            const response = await axios.post(`${this.baseURL}/chat/completions`, {
                model,
                messages,
                temperature
            }, {
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json'
                }
            });

            return response.data.choices[0].message.content;
        } catch (error: any) {
            throw new Error(`AI Client Error: ${error.message}`);
        }
    }

    async createEmbedding(input: string, model: string = 'nomic-embed-text'): Promise<number[]> {
        try {
            const response = await axios.post(`${this.baseURL}/embeddings`, {
                model,
                input
            }, {
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json'
                }
            });

            return response.data.data[0].embedding;
        } catch (error: any) {
            throw new Error(`Embedding Error: ${error.message}`);
        }
    }
}
