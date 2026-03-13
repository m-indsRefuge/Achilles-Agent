export const CHAT_PROMPT = `You are Achilles, an advanced AI development agent.
Answer the following user question based on the provided context.

Context:
{context}

Question:
{question}

Answer:`;

export function fillTemplate(template: string, variables: Record<string, string>): string {
    let result = template;
    for (const [key, value] of Object.entries(variables)) {
        result = result.replace(new RegExp(`\\{${key}\\}`, 'g'), value);
    }
    return result;
}
