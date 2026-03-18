import * as vscode from 'vscode';

export class ProgressManager {
    /**
     * Runs a task with a visible progress bar in the VS Code UI.
     */
    public static async withProgress<T>(
        title: string,
        task: (progress: vscode.Progress<{ message?: string; increment?: number }>) => Promise<T>
    ): Promise<T> {
        return vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: title,
            cancellable: false
        }, async (progress) => {
            try {
                return await task(progress);
            } catch (error: any) {
                this.showError(`Task "${title}" failed: ${error.message}`);
                throw error;
            }
        });
    }

    public static showError(message: string) {
        vscode.window.showErrorMessage(`Achilles: ${message}`);
    }

    public static showInfo(message: string) {
        vscode.window.showInformationMessage(`Achilles: ${message}`);
    }
}
