const DeepLearn = require('deeplearn')
const { Graph, Scalar, Array1D, Array2D, Array3D, Array4D, version } = DeepLearn

module.exports = {
  createGraph: () => {
    const graph = new Graph()

    const t1_s = Scalar.new([1], 'bool')
    const t2_s = Scalar.new([2], 'int32')
    const t1_1d = Array1D.new([1, 2, 3, 4])
    const t2_1d = Array1D.new([5, 6, 7, 8])
    const t1_2d = Array2D.new([2, 2], [1, 2, 3, 4])
    const t2_2d = Array2D.new([2, 2], [5, 6, 7, 8])
    const t1_3d = Array3D.new([2, 2, 1], [1, 2, 3, 4])
    const t2_3d = Array3D.new([2, 2, 1], [5, 6, 7, 8])
    const t1_4d = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4])
    const t2_4d = Array4D.new([2, 2, 1, 1], [5, 6, 7, 8])

    const addNode = (type, ...args) => {
      if (!graph[type]) {
        console.error(`graph.${type} not defined in DeepLearnJs v${version}`)
        return
      }
      graph[type].apply(graph, args)
    }

    addNode('add', t1_3d, t2_3d )
    addNode('argmax', t2_4d )
    addNode('argmaxEquals', t1_3d, t2_3d )
    addNode('concat1d', t1_1d, t2_1d )
    addNode('concat2d', t1_2d, t2_2d, 1 )
    addNode('concat3d', t1_3d, t2_3d, 2 )
    addNode('concat4d', t1_4d, t2_4d, 3 )
    addNode('constant', t1_1d )
    addNode('conv2d', t1_3d, t1_4d, t2_1d, 1, 1, 1, 0 )
    addNode('divide', t1_4d, t2_4d )
    addNode('elu', t1_3d )
    addNode('exp', t1_3d )
    addNode('fusedLinearCombination', t1_4d, t2_4d, t1_s, t2_s )
    addNode('leakyRelu', t1_3d, 3.2 )
    addNode('log', t1_3d )
    addNode('matmul', t1_2d, t2_2d )
    addNode('maxPool', t1_3d, 1, 1 )
    addNode('meanSquaredCost', t1_2d, t2_2d )
    addNode('multiply', t1_3d, t2_3d )
    addNode('placeholder','test', [1,2])
    addNode('prelu', t1_3d, t1_3d)
    addNode('reduceSum', t1_4d )
    addNode('relu', t1_3d )
    addNode('reshape', t2_4d, [4] )
    addNode('sigmoid', t1_3d )
    addNode('softmax', t2_1d )
    addNode('softmaxCrossEntropyCost', t1_3d, t2_3d )
    addNode('square', t2_4d )
    addNode('subtract', t1_3d, t2_3d )
    addNode('tanh', t1_3d )
    addNode('variable','some', t2_4d)

    return graph
  },

  getMissingCoverage: (graph, nodes) => {
    const coverage = {
      addnodeandreturnoutput: true,
      getnodes: true,
    }

    const nameDiff = {
      convolution2d: 'conv2d',
    }

    for (let v in graph) {
      if (typeof graph[v] !== 'function') continue
      v = v.toLowerCase()
      if (!coverage[v]) coverage[v] = false
    }

    for (let n of nodes) {
      let v = n.type.replace(/Node$/,'').toLowerCase()
      if (nameDiff[v]) v = nameDiff[v]
      if (coverage[v] === undefined) throw new Error(`undefined function ${v}`)
      coverage[v] = true
    }

    const noCoverage = []

    for (let v in coverage) {
      if (!coverage[v]) noCoverage.push(v)
    }

    return noCoverage
  }
}
