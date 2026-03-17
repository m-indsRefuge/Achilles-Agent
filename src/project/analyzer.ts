import * as vscode from 'vscode';
import * as fs from 'fs';
import { PythonBridge } from '../comms/pythonBridge';
import { ProgressManager } from '../ui/notifications';

export class ProjectAnalyzer {
    private bridge: PythonBridge;

    constructor(bridge: PythonBridge) {
        this.bridge = bridge;
    }

    public async analyzeWorkspace(): Promise<void> {
        await ProgressManager.withProgress('Achilles: Analyzing Project', async (progress) => {
            const files = await vscode.workspace.findFiles(
                '**/*.{ts,js,py,txt,md}',
                '**/node_modules/**'
            );

            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                progress.report({
                    message: `Indexing ${file.fsPath.split('/').pop()} (${i + 1}/${files.length})`,
                    increment: (1 / files.length) * 100
                });
                await this.indexFile(file.fsPath);
            }
        });
    }

    public async indexFile(filePath: string): Promise<void> {
        try {
            // Clear existing entries for this file to prevent duplicates
            await this.bridge.queryMemory('kb_clear_file', { path: filePath });

            const content = fs.readFileSync(filePath, 'utf-8');
            const lines = content.split('\n');

            const chunks: string[] = [];
            const metadatas: any[] = [];

            for (let i = 0; i < lines.length; i += 50) {
                const chunk = lines.slice(i, i + 50).join('\n');
                if (chunk.trim()) {
                    chunks.push(chunk);
                    metadatas.push({ path: filePath, lineStart: i });
                }
            }

            if (chunks.length > 0) {
                // To keep indexFile atomic but efficient, we still use a small batch internal to the file
                const BATCH_SIZE = 20;
                for (let j = 0; j < chunks.length; j += BATCH_SIZE) {
                    await this.bridge.queryMemory('kb_add_batch', {
                        texts: chunks.slice(j, j + BATCH_SIZE),
                        metadatas: metadatas.slice(j, j + BATCH_SIZE)
                    });
                }
            }
        } catch (error) {
            console.error(`Error indexing file ${filePath}:`, error);
        }
    }
}
