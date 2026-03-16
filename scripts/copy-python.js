const fs = require('fs');
const path = require('path');

function copyRecursiveSync(src, dest) {
    const exists = fs.existsSync(src);
    const stats = exists && fs.statSync(src);
    const isDirectory = exists && stats.isDirectory();
    if (isDirectory) {
        if (!fs.existsSync(dest)) {
            fs.mkdirSync(dest, { recursive: true });
        }
        fs.readdirSync(src).forEach((childItemName) => {
            copyRecursiveSync(path.join(src, childItemName), path.join(dest, childItemName));
        });
    } else {
        fs.copyFileSync(src, dest);
    }
}

const rootDir = path.join(__dirname, '..');
const outDir = path.join(rootDir, 'out');
const srcDir = path.join(rootDir, 'src');

if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
}

// Copy Python files
fs.readdirSync(srcDir).forEach(file => {
    if (file.endsWith('.py')) {
        fs.copyFileSync(path.join(srcDir, file), path.join(outDir, file));
    }
});

// Copy memory_layer
const memoryLayerSrc = path.join(srcDir, 'memory_layer');
const memoryLayerDest = path.join(outDir, 'memory_layer');
if (fs.existsSync(memoryLayerSrc)) {
    copyRecursiveSync(memoryLayerSrc, memoryLayerDest);
}

console.log('Python files and memory_layer copied to out/');
