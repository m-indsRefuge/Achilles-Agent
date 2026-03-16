export interface ScoredResult {
    text: string;
    score: number;
    metadata: any;
}

export class ReRanker {
    /**
     * Simple re-ranker that refines search results based on keyword density
     * or other heuristics. Can be expanded to use a Cross-Encoder LLM.
     */
    public async rerank(query: string, results: ScoredResult[]): Promise<ScoredResult[]> {
        const queryTerms = query.toLowerCase().split(/\s+/);

        const reranked = results.map(res => {
            const text = res.text.toLowerCase();
            let termMatches = 0;

            queryTerms.forEach(term => {
                if (text.includes(term)) {
                    termMatches++;
                }
            });

            // Combine original vector score with keyword match score
            const newScore = (res.score * 0.7) + ((termMatches / queryTerms.length) * 0.3);
            return { ...res, score: newScore };
        });

        return reranked.sort((a, b) => b.score - a.score);
    }
}
