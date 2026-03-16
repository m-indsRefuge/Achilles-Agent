import { TaskRunner, Task } from './taskRunner';
import { ModelInterface } from '../ai/codeLlama/modelInterface';

export interface WorkflowStep {
    description: string;
    task?: Task;
}

export class AutomationManager {
    private taskRunner: TaskRunner;
    private model: ModelInterface;

    constructor(taskRunner: TaskRunner, model: ModelInterface) {
        this.taskRunner = taskRunner;
        this.model = model;
    }

    public async runWorkflow(goal: string): Promise<string[]> {
        const results: string[] = [];
        let currentStatus = `Starting workflow for goal: ${goal}`;
        results.push(currentStatus);

        // Simple 3-step fixed loop for the skeleton
        for (let i = 1; i <= 3; i++) {
            const prompt = `Plan the next step for this goal: ${goal}. Current status: ${currentStatus}.
            Respond ONLY with a JSON task or 'DONE' if finished.
            Example: RUN_TASK: {"type": "shell", "command": "ls"}`;

            const response = await this.model.generate(prompt);

            if (response.includes('DONE')) {
                results.push('Workflow completed successfully.');
                break;
            }

            if (response.includes('RUN_TASK:')) {
                const taskJson = response.split('RUN_TASK:')[1].trim();
                try {
                    const task = JSON.parse(taskJson);
                    const taskResult = await this.taskRunner.run(task);
                    currentStatus = `Step ${i} result: ${taskResult}`;
                    results.push(currentStatus);
                } catch (e: any) {
                    results.push(`Step ${i} failed: ${e.message}`);
                    break;
                }
            } else {
                results.push(`Unexpected response: ${response}`);
                break;
            }
        }

        return results;
    }
}
