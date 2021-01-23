const { createReadStream, writeFile } = require('fs');
const csv = require('csv-parser');
const { createObjectCsvWriter } = require('csv-writer');


module.exports = {
    readCSV: path => new Promise((resolve, reject) => {
        let data = []
        createReadStream(path)
            .on('error', e => reject(e.message))
            .pipe(csv())
            .on('data', row => {
                data.push(row)
            })
            .on('end', () => {
                resolve(data)
            })
            .on('error', reject)
    }),

    writeCSV: (data, path, append = false) => new Promise((resolve, reject) => {
        const writer = createObjectCsvWriter({
            path,
            header: Object.keys(data[0]).map(d => ({id: d, title: d})),
            append
        })

        resolve(writer.writeRecords(data))
    }),

    writeJSON: (data, path) => new Promise((resolve, reject) => {
        writeFile(path, JSON.stringify(data), err => {
            if (err) {
                reject(err)
            }
            resolve()
        })
    })
}
