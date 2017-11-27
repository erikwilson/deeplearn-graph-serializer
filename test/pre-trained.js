const { runNetwork, deserialNetwork } = require('./helpers/network_util')
const { expect } = require('chai')

describe('When loading a pre-trained network', () => {

  let powerOf2 = deserialNetwork(require('./helpers/trained-powerOf2.json'))

  it('Should have an (almost) exact result', async () => {
    const result = await runNetwork(powerOf2,[128,256,512]).val(0)
    const expected = 1023.9799194335938
    const diff = Math.abs(expected - result)
    expect(diff).to.be.below(0.000001)
  })
})
