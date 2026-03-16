import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface Task {
    type: 'shell' | 'edit';
    command?: string;
    filePath?: string;
    content?: string;
}

export class TaskRunner {
    public async run(task: Task): Promise<string> {
        if (task.type === 'shell' && task.command) {
            const confirmation = await vscode.window.showInformationMessage(
                `Achilles wants to run a shell command: \`${task.command}\`. Do you allow this?`,
                'Allow', 'Deny'
            );
            if (confirmation !== 'Allow') {
                return 'Command execution denied by user.';
            }
            return this.executeShell(task.command);
        } else if (task.type === 'edit' && task.filePath && task.content !== undefined) {
            const confirmation = await vscode.window.showInformationMessage(
                `Achilles wants to edit file: \`${task.filePath}\`. Do you allow this?`,
                'Allow', 'Deny'
            );
            if (confirmation !== 'Allow') {
                return 'File edit denied by user.';
            }
            return this.applyEdit(task.filePath, task.content);
        }
        throw new Error('Invalid task type or missing parameters');
    }

    private async executeShell(command: string): Promise<string> {
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
        const { stdout, stderr } = await execAsync(command, { cwd: workspaceRoot });
        if (stderr) {
            return `Output: ${stdout}\nErrors: ${stderr}`;
        }
        return stdout;
    }

    private async applyEdit(filePath: string, content: string): Promise<string> {
        const uri = vscode.Uri.file(filePath);
        const edit = new vscode.WorkspaceEdit();

        // Replace entire file content for simplicity in this skeleton
        const document = await vscode.workspace.openTextDocument(uri);
        const fullRange = new vscode.Range(
            document.positionAt(0),
            document.positionAt(document.getText().length)
        );

        edit.replace(uri, fullRange, content);
        const success = await vscode.workspace.applyEdit(edit);

        if (success) {
            await document.save();
            return `Successfully updated ${filePath}`;
        } else {
            throw new Error(`Failed to apply edit to ${filePath}`);
        }
    }
}
