import { expect } from 'chai';
import * as sinon from 'sinon';
import axios from 'axios';
import { ChatGPTClient } from '../src/comms/chatGPTClient';

describe('ChatGPTClient', () => {
    let client: ChatGPTClient;
    let axiosPostStub: sinon.SinonStub;

    beforeEach(() => {
        client = new ChatGPTClient('test-key', 'http://test-base');
        axiosPostStub = sinon.stub(axios, 'post');
    });

    afterEach(() => {
        sinon.restore();
    });

    it('should send a chat request and return content', async () => {
        const mockResponse = {
            data: {
                choices: [{ message: { content: 'Hello user!' } }]
            }
        };
        axiosPostStub.resolves(mockResponse);

        const response = await client.chat([{ role: 'user', content: 'Hi' }]);

        expect(response).to.equal('Hello user!');
        expect(axiosPostStub.calledOnce).to.be.true;
        const [url, data] = axiosPostStub.getCall(0).args;
        expect(url).to.equal('http://test-base/chat/completions');
        expect(data.messages[0].content).to.equal('Hi');
    });

    it('should send an embedding request and return vector', async () => {
        const mockResponse = {
            data: {
                data: [{ embedding: [0.1, 0.2, 0.3] }]
            }
        };
        axiosPostStub.resolves(mockResponse);

        const embedding = await client.createEmbedding('test text');

        expect(embedding).to.deep.equal([0.1, 0.2, 0.3]);
        expect(axiosPostStub.calledOnce).to.be.true;
    });

    it('should throw error on API failure', async () => {
        axiosPostStub.rejects(new Error('Network Error'));

        try {
            await client.chat([]);
            expect.fail('Should have thrown error');
        } catch (error: any) {
            expect(error.message).to.contain('AI Client Error');
        }
    });
});
