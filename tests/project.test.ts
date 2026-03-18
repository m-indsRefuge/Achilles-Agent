import { expect } from 'chai';
import * as sinon from 'sinon';
import * as vscode from 'vscode';
import { ProjectAnalyzer } from '../src/project/analyzer';
import { PythonBridge } from '../src/comms/pythonBridge';

describe('ProjectAnalyzer', () => {
    let analyzer: ProjectAnalyzer;
    let bridgeStub: sinon.SinonStubbedInstance<PythonBridge>;

    beforeEach(() => {
        bridgeStub = sinon.createStubInstance(PythonBridge);
        analyzer = new ProjectAnalyzer(bridgeStub as any);
    });

    it('should correctly handle indexing of chunks', async () => {
        // Mock vscode.workspace.findFiles
        const findFilesStub = sinon.stub(vscode.workspace, 'findFiles').resolves([]);

        await analyzer.analyzeWorkspace();

        expect(findFilesStub.calledOnce).to.be.true;
        findFilesStub.restore();
    });
});
