const DeepLearn = require('deeplearn')
const GraphSerializer = require('../../src/lib')

const { Array1D } = DeepLearn
const { InCPUMemoryShuffledInputProviderBuilder } = DeepLearn
const { CostReduction, SGDOptimizer } = DeepLearn
const { Graph, Session, NDArrayMath } = DeepLearn

const math = new NDArrayMath('cpu')

module.exports = {
  createNetwork: function () {
    const graph = new Graph()
    const input = graph.placeholder('input', [3])
    const label = graph.placeholder('label', [1])

    let fullyConnectedLayer = graph.layers.dense('layerIn', input, 3)
    // fullyConnectedLayer = graph.layers.dense('layerHidden1', fullyConnectedLayer, 3)
    const output = graph.layers.dense('layerOut', fullyConnectedLayer, 1)
    graph.variable('output', output)

    const cost = graph.meanSquaredCost(label, output)
    graph.variable('cost', cost)

    return { graph, cost, input, output, label }
  },

  trainNetwork: function ({ graph, cost, input, label }, rounds=1) {
    const learningRate = .000001
    const batchSize = 3
    const session = new Session(graph, math)
    const optimizer = new SGDOptimizer(learningRate)

    const inputs = [
      Array1D.new([1.0, 2.0, 3.0]),
      Array1D.new([10.0, 20.0, 30.0]),
      Array1D.new([100.0, 200.0, 300.0])
    ]

    const labels = [
      Array1D.new([4.0]),
      Array1D.new([40.0]),
      Array1D.new([400.0])
    ]

    const shuffledInputProviderBuilder =
        new InCPUMemoryShuffledInputProviderBuilder([inputs, labels])
    const [inputProvider, labelProvider] =
        shuffledInputProviderBuilder.getInputProviders()

    const feedEntries = [
      {tensor: input, data: inputProvider},
      {tensor: label, data: labelProvider}
    ]

    for (let i=0; i<rounds; i++) {
      math.scope(async () => {
        session.train(cost, feedEntries, batchSize, optimizer, CostReduction.MEAN)
      })
    }
  },

  runNetwork: function ({ graph, input, output }, values) {
    const session = new Session(graph, math)
    const data = Array1D.new(values)
    const feedEntries = [{tensor: input, data}]
    return session.eval(output, feedEntries)
  },

  deserialNetwork: function (json) {
    const deserial = GraphSerializer.jsonToGraph(json)
    return {
      graph: deserial.graph,
      cost: deserial.variables.cost,
      input: deserial.placeholders.input,
      output: deserial.variables.output,
      label: deserial.placeholders.label,
    }
  },
}
