import { PythonBridge } from '../../comms/pythonBridge';

export interface ScoredResult {
    text: string;
    score: number;
    metadata: any;
}

export class ReRanker {
    private bridge: PythonBridge;

    constructor(bridge: PythonBridge) {
        this.bridge = bridge;
    }

    /**
     * Advanced re-ranker that uses a Cross-Encoder via the Python bridge.
     */
    public async rerank(query: string, results: ScoredResult[]): Promise<ScoredResult[]> {
        if (!results || results.length === 0) {
            return [];
        }

        try {
            const reranked = await this.bridge.queryMemory('kb_rerank', {
                text: query,
                results: results
            });
            return reranked;
        } catch (error) {
            console.error('Re-ranking failed, falling back to original results:', error);
            return results;
        }
    }
}
