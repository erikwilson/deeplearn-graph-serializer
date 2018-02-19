const { Scalar, SymbolicTensor, Graph } = require('deeplearn')

//------------------------------------------------------------------------------

function graphToJson( graph, idStartsAtZero=true ) {

  const graphNodes = graph.getNodes()

  let idOffset = 0
  if (idStartsAtZero) idOffset = graphNodes[0].id

  const dataToJson = (data) => {
    if (data instanceof SymbolicTensor) return {id:(data.id-idOffset)}
    const { shape, dtype } = data
    const values = Array.from(data.getValues())
    return { values, shape, dtype }
  }

  const jsonNodes = []
  for (let node of graphNodes) {
    const jsonNode = { type: node.constructor.name }
    if (node.data) jsonNode.data = dataToJson(node.data)

    for (let v in node) {
      if (node[v] instanceof Object) continue
      if (v === 'id') continue
      jsonNode[v] = node[v]
    }

    if (Object.keys(node.inputs).length > 0) {
      jsonNode.inputs = {}
      for (let v in node.inputs) {
        jsonNode.inputs[v] = dataToJson(node.inputs[v])
      }
    }
    jsonNode.output = dataToJson(node.output)
    jsonNode.output.shape = node.output.shape
    jsonNodes.push(jsonNode)
  }
  return jsonNodes
}

//------------------------------------------------------------------------------

function jsonToGraph( nodes, tensors={} ) {

  const graph = new Graph()
  const placeholders = {}
  const variables = {}

  for (let node of nodes) {
    const { name, type, inputs, output } = node

    const getTensor = (info) => {
      const { id, dtype, shape } = info
      if (id !== undefined) {
        if (tensors[id] !== undefined) return tensors[id]
        throw new Error(`tensor ${id} not defined`)
      }

      let { values } = info
      if (dtype === 'float32') values = Float32Array.from(values)
      if (dtype === 'int32') values = Int32Array.from(values)
      if (dtype === 'bool') values = Uint8Array.from(values)

      return Scalar.make( shape, {values}, dtype )
    }

    const gFunc = {
      AddNode: () => {
        const t1 = getTensor(inputs.t1)
        const t2 = getTensor(inputs.t2)
        tensors[output.id] =
          graph.add( t1, t2 )
      },
      ArgMaxNode: () => {
        const x = getTensor(inputs.x)
        tensors[output.id] =
          graph.argmax( x )
      },
      ArgMaxEqualsNode: () => {
        const x1 = getTensor(inputs.x1)
        const x2 = getTensor(inputs.x2)
        tensors[output.id] =
          graph.argmaxEquals( x1, x2 )
      },
      Concat1DNode: () => {
        const x1 = getTensor(inputs.x1)
        const x2 = getTensor(inputs.x2)
        tensors[output.id] =
          graph.concat1d( x1, x2 )
      },
      Concat2DNode: () => {
        const x1 = getTensor(inputs.x1)
        const x2 = getTensor(inputs.x2)
        const { axis } = node
        tensors[output.id] =
          graph.concat2d( x1, x2, axis )
      },
      Concat3DNode: () => {
        const x1 = getTensor(inputs.x1)
        const x2 = getTensor(inputs.x2)
        const { axis } = node
        tensors[output.id] =
          graph.concat3d( x1, x2, axis )
      },
      Concat4DNode: () => {
        const x1 = getTensor(inputs.x1)
        const x2 = getTensor(inputs.x2)
        const { axis } = node
        tensors[output.id] =
          graph.concat4d( x1, x2, axis )
      },
      ConstantNode: () => {
        const data = getTensor(node.data)
        tensors[output.id] =
          graph.constant( data )
      },
      Convolution2DNode: () => {
        const x = getTensor(inputs.x)
        const w = getTensor(inputs.w)
        const b = getTensor(inputs.b)
        const { fieldSize, outputDepth, stride, zeroPad } = node
        tensors[output.id] =
          graph.conv2d( x, w, b, fieldSize, outputDepth, stride, zeroPad )
      },
      DivideNode: () => {
        const t1 = getTensor(inputs.t1)
        const t2 = getTensor(inputs.t2)
        tensors[output.id] =
          graph.divide( t1, t2 )
      },
      EluNode: () => {
        const x = getTensor(inputs.x)
        tensors[output.id] =
          graph.elu( x )
      },
      ExpNode: () => {
        const x = getTensor(inputs.x)
        tensors[output.id] =
          graph.exp( x )
      },
      FusedLinearCombinationNode: () => {
        const t1 = getTensor(inputs.t1)
        const t2 = getTensor(inputs.t2)
        const c1 = getTensor(inputs.c1)
        const c2 = getTensor(inputs.c2)
        tensors[output.id] =
          graph.fusedLinearCombination( t1, t2, c1, c2 )
      },
      LeakyReLUNode: () => {
        const x = getTensor(inputs.x)
        const { alpha } = node
        tensors[output.id] =
          graph.leakyRelu( x, alpha )
      },
      LogNode: () => {
        const x = getTensor(inputs.x)
        tensors[output.id] =
          graph.log( x )
      },
      MatMulNode: () => {
        const x1 = getTensor(inputs.x1)
        const x2 = getTensor(inputs.x2)
        tensors[output.id] =
          graph.matmul( x1, x2 )
      },
      MaxPoolNode: () => {
        const x = getTensor(inputs.x)
        const { fieldSize, stride, zeroPad } = node
        tensors[output.id] =
          graph.maxPool( x, fieldSize, stride, zeroPad )
      },
      MeanSquaredCostNode: () => {
        const label = getTensor(inputs.label)
        const prediction = getTensor(inputs.prediction)
        tensors[output.id] =
          graph.meanSquaredCost( label, prediction )
      },
      MultiplyNode: () => {
        const t1 = getTensor(inputs.t1)
        const t2 = getTensor(inputs.t2)
        tensors[output.id] =
          graph.multiply( t1, t2 )
      },
      PlaceholderNode: () => {
        const { shape } = output
        tensors[output.id] = placeholders[name] =
          graph.placeholder( name, shape )
      },
      PReLUNode: () => {
        const x = getTensor(inputs.x)
        const alpha = getTensor(inputs.alpha)
        tensors[output.id] =
          graph.prelu( x, alpha )
      },
      ReduceSumNode: () => {
        const x = getTensor(inputs.x)
        tensors[output.id] =
          graph.reduceSum( x )
      },
      ReLUNode: () => {
        const x = getTensor(inputs.x)
        tensors[output.id] =
          graph.relu( x )
      },
      ReshapeNode: () => {
        const x = getTensor(inputs.x)
        const { shape } = output
        tensors[output.id] =
          graph.reshape( x, shape )
      },
      SigmoidNode: () => {
        const x = getTensor(inputs.x)
        tensors[output.id] =
          graph.sigmoid( x )
      },
      SoftmaxNode: () => {
        const x = getTensor(inputs.x)
        tensors[output.id] =
          graph.softmax( x )
      },
      SoftmaxCrossEntropyCostNode: () => {
        const x = getTensor(inputs.x)
        const target = getTensor(inputs.target)
        tensors[output.id] =
          graph.softmaxCrossEntropyCost( x, target )
      },
      SquareNode: () => {
        const x = getTensor(inputs.x)
        tensors[output.id] =
          graph.square( x )
      },
      SubtractNode: () => {
        const t1 = getTensor(inputs.t1)
        const t2 = getTensor(inputs.t2)
        tensors[output.id] =
          graph.subtract( t1, t2 )
      },
      TanHNode: () => {
        const x = getTensor(inputs.x)
        tensors[output.id] =
          graph.tanh( x )
      },
      VariableNode: () => {
        const data = getTensor(node.data)
        variables[name] = data
        tensors[output.id] =
          graph.variable( name, data )
      },
    }

    if (gFunc[type]) gFunc[type]()
    else throw new Error(`unable to unserialize node type ${type}`)
  }
  return { graph, placeholders, variables, tensors }
}

//------------------------------------------------------------------------------

module.exports = { graphToJson, jsonToGraph }
