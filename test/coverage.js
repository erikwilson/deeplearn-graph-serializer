const GraphSerializer = require('../src/lib')
const { createGraph, getMissingCoverage } = require('./helpers/coverage_util')
const { expect } = require('chai')

describe('DeepLearnJs Graph Coverage', () => {

  const graph = createGraph()
  const serial = GraphSerializer.graphToJson(graph)
  const clone = GraphSerializer.jsonToGraph(serial)
  const newSerial = GraphSerializer.graphToJson(clone.graph)

  it('Should serialize nodes', () => {
    expect(serial.length > 0).to.be.true
  })

  it('Should also deserialize', () => {
    expect(clone.graph).to.not.be.undefined
  })

  it('Should have equal reserialized graphs', () => {
    expect(newSerial).to.deep.equal(serial)
  })

  it('Should have different reserialized graphs if id preserved', () => {
    const serial = GraphSerializer.graphToJson(graph, false)
    const newSerial = GraphSerializer.graphToJson(clone.graph, false)
    expect(newSerial).to.not.deep.equal(serial)
  })

  it('Should cover all of the graph functions', () => {
    expect([]).to.deep.equal(getMissingCoverage(graph,serial))
  })

  it('Should throw an error for unknown node types', () => {
    let error = undefined
    try { GraphSerializer.jsonToGraph([{type:'UnknownNode'}]) }
    catch (err) { error = err }
    expect(error).to.not.be.undefined
  })
})
