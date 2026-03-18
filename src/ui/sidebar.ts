import * as vscode from 'vscode';

/**
 * Manages the state and communication for the Achilles Sidebar.
 */
export class SidebarManager {
    private isThinking: boolean = false;

    public setThinking(thinking: boolean) {
        this.isThinking = thinking;
        // In a real implementation, this would post a message to the Webview
        // to show/hide a loading spinner.
    }

    public getStatus(): string {
        return this.isThinking ? 'Achilles is thinking...' : 'Achilles is ready.';
    }
}
