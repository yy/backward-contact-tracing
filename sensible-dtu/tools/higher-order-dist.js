const ArgumentParser = require('argparse').ArgumentParser;
const { sum } = require('d3-array');
const io = require('../src/io');


function g(d, order) {
    if (order === 1) {
        return d.x * d.y
    }
    if (order === 2) {
        return d.x * (d.x - 1) * d.y
    }
}


// Read arguments.
const parser = new ArgumentParser({
    version: '0.0.1',
    addHelp: true,
    description: 'Tool to calculate the higher order CCDF from a first order degree distribution.'
});
parser.addArgument(['--input'], {
    dest: 'input',
    required: true,
    help: 'Path to the CSV file containing the original degree distribution.'
});
parser.addArgument(['--output'], {
    dest: 'output',
    required: true,
    help: 'Path to the CSV file where the results are saved.'
});
parser.addArgument(['--order'], {
    dest: 'order',
    type: 'int',
    defaultValue: 1,
    help: 'Order to calculate from the distribution [1].'
});
args = parser.parseArgs();

// Run calculations.
(async () => {
    // Load original distribution.
    let ccdf = await io.readCSV(args.input)
    ccdf = ccdf.map(d => ({
        x: +d.x,
        y: +d.y
    }))

    // Calculate p(k).
    let p = ccdf.map((d, i) => ({
        x: i > 0 ? d.x : 0,
        y: i > 0 ? ccdf[i - 1].y - d.y : 1 - d.y
    }))

    // Convert to higher order distribution.
    let q = p.map(d => ({
        x: d.x,
        y: g(d, args.order)
    }))
    let qSum = sum(q, d => d.y)

    // Normalize q(k).
    q = q.map(d => ({
        x: d.x,
        y: d.y / qSum
    }))

    // Save new CCDF.
    let s = 1
    const ccdfQ = q.map(d => {
        s -= d.y
        return {
            x: d.x,
            y: s
        }
    })
    io.writeCSV(ccdfQ, args.output)
})()
