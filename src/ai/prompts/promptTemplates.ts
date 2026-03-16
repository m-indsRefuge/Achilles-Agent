export const CHAT_PROMPT = `You are Achilles, an advanced AI development agent.
Answer the following user question based on the provided context.

Context:
{context}

Question:
{question}

If you need to perform an action, you can suggest a task in JSON format using the prefix RUN_TASK:.
Example: RUN_TASK: {"type": "shell", "command": "npm test"}
Example: RUN_TASK: {"type": "edit", "filePath": "src/main.ts", "content": "console.log('hello');"}

Answer:`;

export function fillTemplate(template: string, variables: Record<string, string>): string {
    let result = template;
    for (const [key, value] of Object.entries(variables)) {
        result = result.replace(new RegExp(`\\{${key}\\}`, 'g'), value);
    }
    return result;
}
