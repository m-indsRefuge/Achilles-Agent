import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';

export class PythonBridge {
    private pythonProcess: ChildProcess | null = null;
    private pythonPath: string;
    private scriptPath: string;

    constructor(pythonPath: string = 'python3') {
        this.pythonPath = pythonPath;
        // Resolve script path relative to the workspace root to be more robust
        // In a real VS Code extension, this would use context.extensionPath
        this.scriptPath = path.resolve(__dirname, '..', 'bridge_entry.py');
        if (__dirname.endsWith('out/comms')) {
            // If running from compiled 'out' directory
            this.scriptPath = path.resolve(__dirname, '..', '..', 'src', 'bridge_entry.py');
        }
    }

    public async queryMemory(layer: string, query: any): Promise<any> {
        return new Promise((resolve, reject) => {
            const pythonSrcDir = path.resolve(path.dirname(this.scriptPath));
            const child = spawn(this.pythonPath, [
                this.scriptPath,
                layer,
                JSON.stringify(query)
            ], {
                env: { ...process.env, PYTHONPATH: pythonSrcDir }
            });

            let output = '';
            let error = '';

            child.stdout?.on('data', (data: Buffer) => {
                output += data.toString();
            });

            child.stderr?.on('data', (data: Buffer) => {
                error += data.toString();
            });

            child.on('close', (code: number) => {
                if (code !== 0) {
                    reject(new Error(`Python process exited with code ${code}: ${error}`));
                } else {
                    try {
                        resolve(JSON.parse(output));
                    } catch (e) {
                        reject(new Error(`Failed to parse Python output: ${output}`));
                    }
                }
            });
        });
    }
}
