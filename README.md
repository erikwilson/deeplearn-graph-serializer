![travis-ci](https://api.travis-ci.org/erikwilson/deeplearn-graph-serializer.svg)
# deeplearn-graph-serializer
**WARNING:** This is an unofficial serialization/deserialization library for DeepLearnJs graphs. It may be fragile and break with future revisions.

## Usage
Require or import this module in your code:
```js
const GraphSerializer = require('deeplearn-graph-serializer')
```
After a graph is created its structure and values can be exported using `GraphSerializer.graphToJson`:
```js
const graphJson = GraphSerializer.graphToJson(graph)
console.log(JSON.stringify(graphJson))
```
A network can be recreated from JSON using `GraphSerializer.jsonToGraph`:
```js
const net = GraphSerializer.jsonToGraph(graphJson)
const { graph, placeholders, variables, tensors } = net
```
The `GraphSerializer.jsonToGraph` method returns a graph object, any tensors created, placeholder references to tensors by name, and variable data by name.

## Advanced Usage
The `GraphSerializer.graphToJson` method normally returns a JSON that references tensors starting from id `0`. To preserve the tensor id we can pass `false` as the second parameter to `GraphSerializer.graphToJson`. In this way it is possible to serialize a collection of graphs that reference tensors outside of their own graph:

```js
const graph1 = new Graph()
const v = graph1.variable('v',Scalar.new(2))
const graph2 = new Graph()
const result = graph2.multiply(v,v)
graph2.variable('result',result)

const graph1Json = GraphSerializer.graphToJson(graph1, false)
const graph2Json = GraphSerializer.graphToJson(graph2, false)
```

Then when using `GraphSerializer.jsonToGraph` we would pass a tensor object as the second parameter, this allows the deserializer to reference tensors from other deserialized graphs:

```js
let deserial =  GraphSerializer.jsonToGraph(graph1Json)
deserial.variables.v.set(3)
deserial =  GraphSerializer.jsonToGraph(graph2Json, deserial.tensors)
const session = new Session(deserial.graph, math)
const squared = await session.eval(deserial.variables.result).val(0)
console.log(squared) // = 9
```
