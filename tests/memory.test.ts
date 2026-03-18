import { expect } from 'chai';
import * as sinon from 'sinon';
import { ReRanker } from '../src/ai/reRanker/reRankerInterface';
import { PythonBridge } from '../src/comms/pythonBridge';

describe('ReRanker', () => {
    let reranker: ReRanker;
    let bridgeStub: sinon.SinonStubbedInstance<PythonBridge>;

    beforeEach(() => {
        bridgeStub = sinon.createStubInstance(PythonBridge);
        reranker = new ReRanker(bridgeStub as any);
    });

    it('should return empty list if results are empty', async () => {
        const results = await reranker.rerank('test', []);
        expect(results).to.be.empty;
    });

    it('should call Python bridge for re-ranking', async () => {
        const mockResults = [
            { text: 'A', score: 0.1, metadata: {} },
            { text: 'B', score: 0.2, metadata: {} }
        ];
        const mockReranked = [
            { text: 'B', score: 0.9, metadata: {} },
            { text: 'A', score: 0.8, metadata: {} }
        ];

        bridgeStub.queryMemory.resolves(mockReranked);

        const results = await reranker.rerank('query', mockResults);

        expect(results[0].text).to.equal('B');
        expect(bridgeStub.queryMemory.calledWith('kb_rerank', sinon.match.any)).to.be.true;
    });

    it('should fallback to original results on bridge error', async () => {
        const mockResults = [{ text: 'A', score: 0.1, metadata: {} }];
        bridgeStub.queryMemory.rejects(new Error('Bridge Failure'));

        const results = await reranker.rerank('query', mockResults);

        expect(results).to.deep.equal(mockResults);
    });
});
