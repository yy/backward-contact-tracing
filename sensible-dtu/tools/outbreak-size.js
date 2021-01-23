const ArgumentParser = require('argparse').ArgumentParser;
const io = require('../src/io');
const log = require('../src/log');
const { nest } = require('d3-collection')
const { median, quantile, max, min } = require('d3-array')


// Read arguments.
const parser = new ArgumentParser({
    version: '0.0.1',
    addHelp: true,
    description: 'Tool to calculate outbreak size from simulation logs.'
});
parser.addArgument(['--logs'], {
    dest: 'logs',
    required: true,
    help: 'Path to the CSV file containing the event logs.'
});
parser.addArgument(['--output'], {
    dest: 'output',
    required: true,
    help: 'Path to the CSV file where the results are saved.'
});
const args = parser.parseArgs();


// Calculate epidemiology curves.
(async () => {
    // Read logs.
    const logs = await io.readCSV(args.logs)

    // Calculate number of infected.
    const data = nest()
        .key(d => d.id)
        .rollup(values => values.filter(d => d.type === 'i').length)
        .entries(logs)
        .map(d => d.value)
        .sort((a, b) => a - b)
    let q1 = quantile(data, 0.25)
    let q3 = quantile(data, 0.75)
    let iqr = q3 - q1
    let outliers = data.filter(d => d < q1 - 1.5 * iqr || d > q3 + 1.5 * iqr)
    const result = {
        median: median(data),
        q1, q3,
        whiskers: {
            lower: min(data.filter(d => d > q1 - 1.5 * iqr)),
            upper: max(data.filter(d => d < q3 + 1.5 * iqr))
        },
        outliers: (() => {
            const extreme = []
            const mild = []
            outliers.map(d => {
                if (d < q1 - 3 * iqr || d > q3 + 3 * iqr) {
                    extreme.push(d)
                } else {
                    mild.push(d)
                }
            })
            return {mild, extreme}
        })()
    }

    // Save results.
    io.writeJSON(result, args.output)
})()
