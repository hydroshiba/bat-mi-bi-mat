import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

env.allowLocalModels = false;

let tokenizer = null;
let quadgramProbs = null;
let rankerParams = null;

async function init() {
    const btn = document.getElementById("solveBtn");
    btn.disabled = true;
    
    try {
        const [qRes, rRes] = await Promise.all([
            fetch('json/quadgram.json'),
            fetch('json/ranker_params.json')
        ]);
        quadgramProbs = await qRes.json();
        rankerParams = await rRes.json();
        
        tokenizer = await AutoTokenizer.from_pretrained('NlpHUST/gpt2-vietnamese');
        
        btn.disabled = false;
        btn.innerText = "Giải!";
        btn.addEventListener('click', generateCiphers);
    } catch (e) {
        console.error("Error loading resources:", e);
        btn.innerText = "Error Loading!";
    }
}
init();

function caesarCipher(text, shift) {
    let result = "";
    for (let i = 0; i < text.length; i++) {
        let char = text[i];
        if (/[a-zA-Z]/.test(char)) {
            let isUpper = char === char.toUpperCase();
            let base = isUpper ? 65 : 97;
            let shifted = String.fromCharCode(((char.charCodeAt(0) - base + shift) % 26) + base);
            result += shifted;
        } else {
            result += char;
        }
    }
    return result;
}

function atbashCipher(text) {
    let result = "";
    for (let i = 0; i < text.length; i++) {
        let char = text[i];
        if (/[a-zA-Z]/.test(char)) {
            let isUpper = char === char.toUpperCase();
            let base = isUpper ? 65 : 97;
            let reverseBase = isUpper ? 90 : 122; // Z or z
            let shifted = String.fromCharCode(reverseBase - (char.charCodeAt(0) - base));
            result += shifted;
        } else {
            result += char;
        }
    }
    return result;
}

function flattenText(str) {
    return str.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/đ/g, "d").replace(/Đ/g, "D");
}

function quadgramLoss(text) {
    let loss = 0.0;
    let nQuadgrams = 0;
    for (let i = 0; i < text.length - 3; i++) {
        let quad = text.substring(i, i + 4);
        let prob = quadgramProbs[quad] !== undefined ? quadgramProbs[quad] : 1e-10;
        loss -= Math.log(prob);
        nQuadgrams++;
    }
    return nQuadgrams > 0 ? loss / nQuadgrams : Infinity;
}

function vowelPercentage(text) {
    const vowels = 'aeiou';
    let count = 0;
    for (let i = 0; i < text.length; i++) {
        if (vowels.includes(text[i].toLowerCase())) {
            count++;
        }
    }
    return text.length > 0 ? count / text.length : 0;
}

async function scoreCandidatesBatch(candidates) {
    if (!rankerParams || !quadgramProbs) {
        candidates.forEach(c => c.score = 0);
        return;
    }

    const texts = candidates.map(c => c.text);
    const qLosses = texts.map(t => quadgramLoss(t));
    const vPercs = texts.map(t => vowelPercentage(t));
    let tokenCounts = new Array(candidates.length).fill(0);

    if (tokenizer) {
        // Tokenizing 52 strings one by one using Promise.all was causing issues.
        // Let's tokenize them efficiently but safely.
        try {
            for (let i = 0; i < candidates.length; i++) {
                const encoding = await tokenizer(texts[i]);
                tokenCounts[i] = encoding.input_ids.data.length;
            }
        } catch (e) {
            console.error("Tokenizer error:", e);
        }
    }

    const { weights, bias, feature_means, feature_stds } = rankerParams;

    for (let i = 0; i < candidates.length; i++) {
        const normQLoss = (qLosses[i] - feature_means[0]) / (feature_stds[0] + 1e-8);
        const normVPerc = (vPercs[i] - feature_means[1]) / (feature_stds[1] + 1e-8);
        const normTokenCount = (tokenCounts[i] - feature_means[2]) / (feature_stds[2] + 1e-8);
        
        const Z = (weights[0] * normQLoss) + 
                  (weights[1] * normVPerc) + 
                  (weights[2] * normTokenCount) + 
                  bias;
        
        candidates[i].score = 1 / (1 + Math.exp(-Z));
    }
}

async function generateCiphers() {
    let text = document.getElementById("inputText").value;
    const outputDiv = document.getElementById("output");
    outputDiv.innerHTML = "<p>Processing & Ranking...</p>";

    // Give the browser a moment to actually render the "Processing" text before we lock the thread
    await new Promise(resolve => setTimeout(resolve, 50));

    if (!text.trim()) {
        outputDiv.innerHTML = "<p>Please enter some text.</p>";
        return;
    }
    
    text = flattenText(text);

    let candidates = [];
    const alphabet = "abcdefghijklmnopqrstuvwxyz";

    for (let shift = 0; shift < 26; shift++) {
        let shiftedText = caesarCipher(text, shift);
        let mappedAlpha = caesarCipher(alphabet, shift);
        
        // 1. Caesar only
        candidates.push({ 
            text: shiftedText, 
            steps: [
                {
                    label: `Bước 1: Dịch bảng chữ cái sang ${shift} vị trí`,
                    alphaBefore: alphabet,
                    alphaAfter: mappedAlpha,
                    textBefore: text,
                    textAfter: shiftedText
                }
            ]
        });

        // 2. Caesar + Atbash combined
        let combinedText = atbashCipher(shiftedText);
        candidates.push({ 
            text: combinedText, 
            steps: [
                {
                    label: `Bước 1: Dịch bảng chữ cái sang ${shift} vị trí`,
                    alphaBefore: alphabet,
                    alphaAfter: mappedAlpha,
                    textBefore: text,
                    textAfter: shiftedText
                },
                {
                    label: "Bước 2: Đảo ngược bảng chữ cái",
                    alphaBefore: alphabet,
                    alphaAfter: atbashCipher(alphabet),
                    textBefore: shiftedText,
                    textAfter: combinedText
                }
            ]
        });
    }
    
    // Score all candidates securely in a single batch
    await scoreCandidatesBatch(candidates);
    
    // Rank descending
    candidates.sort((a, b) => b.score - a.score);

    let resultsHTML = "";
    for (let i = 0; i < candidates.length; i++) {
        const c = candidates[i];

        // Map score to a gradient ratio (1.0 = Max Score, 0.0 = Min Score)
        let ratio = 1;
        if (candidates.length > 1) {
            const maxScore = candidates[0].score;
            const minScore = candidates[candidates.length - 1].score;
            ratio = maxScore === minScore ? 1 : (c.score - minScore) / (maxScore - minScore);
        }
        
        // hue 120 is green, 0 is red. Desaturated to 30% for a grayish look, 60% lightness.
        const hue = Math.round(120 * ratio);
        const color = `hsl(${hue}, 30%, 60%)`;
        
        const percentageText = (c.score * 100).toFixed(3);

        resultsHTML += `<div class="result-item"><div class="cipher-text">${c.text}</div><div class="label" style="color: ${color};">#${i + 1} - Độ tự tin: ${percentageText}%</div><button class="solution-btn" data-idx="${i}">Xem cách giải</button></div>`;
    }

    outputDiv.innerHTML = resultsHTML;

    // Attach modal listeners
    const modal = document.getElementById('solutionModal');
    const modalContent = document.getElementById('modalContent');
    const closeModalBtn = document.getElementById('closeModalBtn');
    
    closeModalBtn.addEventListener('click', () => {
        modal.classList.remove('active');
    });

    outputDiv.querySelectorAll('.solution-btn').forEach((btn) => {
        btn.addEventListener('click', (e) => {
            const idx = parseInt(e.target.getAttribute('data-idx'));
            const c = candidates[idx];
            
            let html = "";
            for (let step of c.steps) {
                const alphaBefore = step.alphaBefore.split('').join(' ');
                const alphaAfter = step.alphaAfter.split('').join(' ');
                html += `
                <div class="sol-label">${step.label}:</div>
                <div class="sol-code">${alphaBefore}\n${alphaAfter}</div>
                <div class="sol-label">Áp dụng lên mật thư:</div>
                <div class="sol-code">${step.textBefore}\n${step.textAfter}</div>
                <br>
                `;
            }
            
            modalContent.innerHTML = html;
            modal.classList.add('active');
        });
    });
}
