const ArgumentParser = require('argparse').ArgumentParser;
const { nest } = require('d3-collection');
const { min } = require('d3-array');
const io = require('../src/io');


// Read arguments.
const parser = new ArgumentParser({
    version: '0.0.1',
    addHelp: true,
    description: 'Tool to calculate the CCDF of constrained degrees of specific nodes.'
})
parser.addArgument(['--output'], {
    dest: 'output',
    required: true,
    help: 'Path to the CSV file where the results are saved.'
})
parser.addArgument(['--logs'], {
    dest: 'logs',
    defaultValue: null,
    help: 'Path to the CSV file containing the simulation logs.'
})
parser.addArgument(['--meta'], {
    dest: 'meta',
    defaultValue: null,
    help: 'Path to the CSV file containing the meta data of the runs.'
})
parser.addArgument(['--network'], {
    dest: 'network',
    required: true,
    help: 'Path to the CSV file containing the temporal network.'
})
parser.addArgument(['--duration'], {
    dest: 'duration',
    type: 'int',
    help: 'Minimum total duration for interactions to be accepted as contacts in 5min units [1].'
})
parser.addArgument(['--type'], {
    dest: 'type',
    defaultValue: 'i',
    help: 'Type of nodes from the log events to consider in the degree distribution [i].'
})
parser.addArgument(['--tmax'], {
    dest: 'tmax',
    type: 'int',
    defaultValue: null,
    help: 'Maximum elapsed time to consider log events in the degree distribution [null].'
})
args = parser.parseArgs();


(async () => {
    // Load network.
    //const meta = await io.readCSV(args.meta)
    const network = await io.readCSV(args.network)

    // Copy reversed links to links.
    const links = network.concat(network.map(d => ({n1: d.n2, n2: d.n1})))

    // Count constrained degrees for each node.
    const degrees = nest()
        .key(d => d.n1)
        .rollup(values => nest()
            .key(d => d.n2)
            .rollup(v => v.length)
            .entries(values)
            .map(d => d.value)
        )
        .entries(links)
        // Filter by duration.
        .map(d => ({
            id: +d.key,
            contacts: d.value.filter(c => c >= args.duration)
        }))
        // Convert to an object.
        .reduce((map, d) => Object.assign(map, {[d.id]: d.contacts.length}), {})

    // Collect nodes to build degree distribution over.
    let nodes
    if (args.logs !== null) {
        // If logs are specified, collect node IDs from the log events.
        let logs = await io.readCSV(args.logs)

        // Filter logs.
        logs = logs.filter(d => d.elapsed !== '0')
            .filter(d => d.type === 'e')
        if (args.tmax !== null) {
            logs = logs.filter(d => +d.elapsed <= args.tmax)
        }

        // Collect nodes.
        switch (args.type) {
            default:
            case 'i':
                // Infected nodes.
                nodes = logs.map(d => d.node)
                break
            case 'source':
                // Parent nodes.
                nodes = logs.map(d => d.source)
        }
    } else {
        // If logs are not provided, we just take all nodes from the degrees.
        nodes = Object.keys(degrees)
    }

    // Map nodes to degrees and aggregate distribution.
    // the CCDF can be quickly estimated by sorting the observed values, ranking them and inverting the resulting
    // function.
    let ccdf = nodes.map(d => degrees[d])
        .sort((a, b) => a - b)
        .map((d, i) => ({x: d, y: 1 - i / nodes.length}))

    // Simplify CCDF (reduce rows with the same x values to the last occurrence).
    ccdf = nest()
        .key(d => d.x)
        .rollup(values => min(values, d => d.y))
        .entries(ccdf)
        .map(d => ({
            x: +d.key,
            y: d.value
        }))

    // Save CCDF.
    io.writeCSV(ccdf, args.output)
})()
