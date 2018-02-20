const GraphSerializer = require('../src/lib')
const dl = require('deeplearn')
const { expect } = require('chai')

describe('Creating a graph that uses a variable from another graph', () => {

  const math = new dl.NDArrayMath('cpu')

  const g1 = new dl.Graph()
  const v = g1.variable('v', dl.variable(dl.scalar(2)))
  const g2 = new dl.Graph()
  const result = g2.multiply(v,v)
  g2.variable('result',result)

  it('Should produce correct results', async () => {
    let session = new dl.Session(g2, math)
    let squared = await session.eval(result).val(0)
    expect(squared).to.equal(4)
  })

  it('Should produce an error if not properly de/serialized', () => {
    let error = undefined
    const invalid = GraphSerializer.graphToJson(g2)
    try { GraphSerializer.jsonToGraph(invalid) }
    catch (err) { error = err }
    expect(error).to.not.be.undefined
  })

  it('Should deserialize multiple graphs', async () => {
    let deserial
    const g1Serial = GraphSerializer.graphToJson(g1, false)
    const g2Serial = GraphSerializer.graphToJson(g2, false)
    deserial =  GraphSerializer.jsonToGraph(g1Serial)
    deserial.variables.v.assign(dl.scalar(3))
    deserial =  GraphSerializer.jsonToGraph(g2Serial, deserial.tensors)
    const session = new dl.Session(deserial.graph, math)
    const squared = await session.eval(deserial.variables.result).val(0)
    expect(squared).to.equal(9)
  })
})
