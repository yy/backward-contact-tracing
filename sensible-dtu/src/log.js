const chalk = require('chalk');


module.exports = (() => {
    let _ = {
        start: Date.now()
    }

    function padded (x) {
        return `${Math.floor(x / 10)}${x % 10}`
    }

    function formatElapsedTime (ms) {
        let s = Math.floor(ms / 1000)
        let h = Math.floor(s / 3600)
        let m = Math.floor((s % 3600) / 60)
        return `${padded(h)}:${padded(m)}:${padded(s)}`
    }

    return {
        e: message => console.log(chalk.red(`ERRO [${formatElapsedTime(Date.now() - _.start)}]: ${message}`)),
        i: message => console.log(chalk.white(`INFO [${formatElapsedTime(Date.now() - _.start)}]: ${message}`))
    }
})()
