import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';

export class PythonBridge {
    private child: ChildProcess | null = null;
    private pythonPath: string;
    private scriptPath: string;
    private pendingRequests: Map<string, (value: any) => void> = new Map();
    private readyPromise: Promise<void>;

    constructor(pythonPath: string = 'python3') {
        this.pythonPath = pythonPath;
        this.scriptPath = path.resolve(__dirname, '..', 'bridge_entry.py');
        this.readyPromise = this.start();
    }

    private async start(): Promise<void> {
        return new Promise((resolve, reject) => {
            const pythonSrcDir = path.dirname(this.scriptPath);
            this.child = spawn(this.pythonPath, [this.scriptPath], {
                env: { ...process.env, PYTHONPATH: pythonSrcDir }
            });

            let error = '';

            this.child.stdout?.on('data', (data: Buffer) => {
                const lines = data.toString().split('\n');
                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const message = JSON.parse(line);
                        if (message.status === 'ready') {
                            resolve();
                        } else if (message.requestId) {
                            const resolveRequest = this.pendingRequests.get(message.requestId);
                            if (resolveRequest) {
                                resolveRequest(message.data);
                                this.pendingRequests.delete(message.requestId);
                            }
                        }
                    } catch (e) {
                        console.error('Failed to parse Python output:', line);
                    }
                }
            });

            this.child.stderr?.on('data', (data: Buffer) => {
                error += data.toString();
            });

            this.child.on('close', (code: number) => {
                console.error(`Python process closed with code ${code}: ${error}`);
                this.child = null;
                // Auto-recovery attempt
                setTimeout(() => {
                    console.log('Attempting to restart Python bridge...');
                    this.readyPromise = this.start();
                }, 1000);
            });
        });
    }

    public async queryMemory(layer: string, query: any): Promise<any> {
        await this.readyPromise;
        if (!this.child || !this.child.stdin) throw new Error('Python bridge not running');

        return new Promise((resolve) => {
            const requestId = Math.random().toString(36).substring(7);
            this.pendingRequests.set(requestId, resolve);
            this.child!.stdin!.write(JSON.stringify({ requestId, layer, query }) + '\n');
        });
    }

    public dispose() {
        if (this.child) {
            this.child.kill();
        }
    }
}
