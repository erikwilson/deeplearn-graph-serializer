const GraphSerializer = require('../src/lib')
const {
  createNetwork,
  trainNetwork,
  runNetwork,
  deserialNetwork
} = require('./helpers/network_util')
const { expect } = require('chai')

describe('When setting up a new training graph and clone', () => {

  let { originalNet, originalSerial, cloneNet, cloneSerial } = {}

  before(async () => {
    originalNet = createNetwork()
    await trainNetwork(originalNet)
    originalSerial = GraphSerializer.graphToJson(originalNet.graph)
    cloneNet = deserialNetwork(originalSerial)
    cloneSerial = GraphSerializer.graphToJson(cloneNet.graph)
  })

  it('Should have equal JSON', () => {
    expect(cloneSerial).to.deep.equal(originalSerial)
  })

  it('Should have (almost) equal initial results', async () => {
    const result1 = await runNetwork(originalNet,[0.1,0.2,0.3]).val(0)
    const result2 = await runNetwork(cloneNet,[0.1,0.2,0.3]).val(0)
    const diff = Math.abs(result1 - result2)
    expect(diff).to.be.below(0.000001)
  })

  describe('After training original and clone networks', () => {

    let { result1, result2 } = {}

    before(async () => {
      await trainNetwork(originalNet, 99)
      await trainNetwork(cloneNet, 99)
      result1 = await runNetwork(originalNet,[0.1,0.2,0.3]).val(0)
      result2 = await runNetwork(cloneNet,[0.1,0.2,0.3]).val(0)
    })

    it('Should have (almost) equal trained results', async () => {
      const diff = Math.abs(result1 - result2)
      expect(diff).to.be.below(0.000001)
    })

    it('Should have (almost) correct prediction', async () => {
      const diff = Math.abs(0.4 - result2)
      expect(diff).to.be.below(0.1)
    })

    it('Should have equal recloned JSON', async () => {
      cloneSerial = GraphSerializer.graphToJson(cloneNet.graph)
      const cloneNet2 = deserialNetwork(cloneSerial)
      const cloneSerial2 = GraphSerializer.graphToJson(cloneNet2.graph)
      expect(cloneSerial2).to.deep.equal(cloneSerial)
    })
  })
})
