
/**
 * ============================================================================
 * Padhaku AI - Backend API Server (Node.js + Express + Gemini SDK)
 * ============================================================================
 * Features:
 * 1. Multimodal Quiz Generation (/api/generate-quiz):
 *    Accepts text notes and/or image uploads (via Multer memory storage),
 *    processes them via Gemini 2.5 Flash with Structured JSON Schema output.
 * 2. Revision Vault Persistence (/api/sessions):
 *    Full CRUD endpoints (GET, POST, PATCH, DELETE) backed by a lightweight
 *    JSON file store (with automatic initialization) so sessions persist across restarts.
 * 3. Production Hardening:
 *    CORS support, payload limits, input validation, and structured error handling.
 *
 * Setup & Run:
 *   npm init -y
 *   npm install express cors dotenv multer @google/genai
 *   export GEMINI_API_KEY="your_api_key_here"
 *   node server.js
 * ============================================================================
 */

import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

// ============================================================================
// 1. CONFIGURATION & INITIALIZATION
// ============================================================================
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 3000;
const DATA_DIR = path.join(__dirname, 'data');
const DB_FILE = path.join(DATA_DIR, 'revision_vault.json');

// Initialize the official Google Gen AI SDK (automatically reads GEMINI_API_KEY)
if (!process.env.GEMINI_API_KEY) {
    console.warn('⚠️  WARNING: GEMINI_API_KEY environment variable is not set.');
}
const ai = new GoogleGenAI();

const app = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: '15mb' }));
app.use(express.urlencoded({ extended: true, limit: '15mb' }));

// Configure Multer for in-memory image uploads (max 10MB)
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 10 * 1024 * 1024 },
    fileFilter: (_req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files (PNG, JPG, WEBP) are allowed.'));
        }
    }
});

// ============================================================================
// 2. LIGHTWEIGHT JSON STORAGE HELPER (REVISION VAULT DB)
// ============================================================================
async function ensureStorageReady() {
    try {
        await fs.mkdir(DATA_DIR, { recursive: true });
        try {
            await fs.access(DB_FILE);
        } catch {
            await fs.writeFile(DB_FILE, JSON.stringify([], null, 2), 'utf-8');
        }
    } catch (err) {
        console.error('Storage initialization failed:', err);
    }
}

async function readSessions() {
    await ensureStorageReady();
    const raw = await fs.readFile(DB_FILE, 'utf-8');
    return JSON.parse(raw);
}

async function writeSessions(sessions) {
    await ensureStorageReady();
    await fs.writeFile(DB_FILE, JSON.stringify(sessions, null, 2), 'utf-8');
}

// ============================================================================
// 3. GEMINI STRUCTURED OUTPUT SCHEMA DEFINITION
// ============================================================================
const quizResponseSchema = {
    type: Type.OBJECT,
    properties: {
        title: {
            type: Type.STRING,
            description: 'Concise, descriptive study session title.'
        },
        subject: {
            type: Type.STRING,
            description: 'Academic subject or discipline (e.g., Operating Systems, Biology).'
        },
        summary: {
            type: Type.STRING,
            description: 'A crisp 2-3 sentence revision summary of the material.'
        },
        keyTakeaways: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: '4 high-yield bullet takeaways for rapid revision.'
        },
        questions: {
            type: Type.ARRAY,
            items: {
                type: Type.OBJECT,
                properties: {
                    question: { type: Type.STRING },
                    options: {
                        type: Type.ARRAY,
                        items: { type: Type.STRING },
                        description: 'Exactly 4 distinct multiple-choice options.'
                    },
                    correctIndex: {
                        type: Type.INTEGER,
                        description: '0-based index (0 to 3) of the single best answer.'
                    },
                    explanation: {
                        type: Type.STRING,
                        description: 'Clear educational explanation citing why the correct answer is right.'
                    }
                },
                required: ['question', 'options', 'correctIndex', 'explanation']
            }
        }
    },
    required: ['title', 'subject', 'summary', 'keyTakeaways', 'questions']
};

// ============================================================================
// 4. API ROUTES
// ============================================================================

/**
 * Health Check Endpoint
 */
app.get('/api/health', (_req, res) => {
    res.json({
        status: 'ok',
        service: 'Padhaku AI Backend',
        timestamp: new Date().toISOString()
    });
});

/**
 * POST /api/generate-quiz
 * Generates an interactive study quiz & cheat-sheet from notes and/or an image.
 * Supports both `multipart/form-data` (file field: "image") and `application/json` (base64).
 */
app.post('/api/generate-quiz', upload.single('image'), async (req, res) => {
    try {
        const notesText = (req.body.notes || '').trim();
        const numQuestions = Math.min(Math.max(parseInt(req.body.numQuestions, 10) || 5, 1), 15);
        const difficulty = req.body.difficulty || 'Exam-Ready';

        // Support either Multer file buffer OR base64 payload from JSON
        let imageBase64 = req.body.imageBase64 || null;
        let imageMime = req.body.imageMime || null;

        if (req.file) {
            imageBase64 = req.file.buffer.toString('base64');
            imageMime = req.file.mimetype;
        }

        if (!notesText && !imageBase64) {
            return res.status(400).json({
                error: 'Please provide study notes text or upload a study material image.'
            });
        }

        const systemInstruction = `You are Padhaku AI, an elite academic tutor and exam-prep architect.
Analyze the provided study image and/or text notes carefully.
1. Identify the core subject and create a concise, descriptive session title.
2. Write a crisp 2-3 sentence revision summary and extract 4 high-yield bullet takeaways.
3. Generate exactly ${numQuestions} multiple-choice questions at a "${difficulty}" difficulty level.
4. Each question MUST have exactly 4 distinct options, the 0-based index of the single best answer (0 to 3), and a clear, educational explanation citing why the correct answer is right.`;

        // Construct multimodal contents array
        const contents = [];
        if (notesText) {
            contents.push(`Study Material / Notes:\n${notesText}`);
        } else {
            contents.push(
                `Analyze the attached study material image and generate a comprehensive revision summary and ${numQuestions}-question quiz.`
            );
        }

        if (imageBase64 && imageMime) {
            contents.push({
                inlineData: {
                    data: imageBase64,
                    mimeType: imageMime
                }
            });
        }

        // Call Gemini 2.5 Flash with Structured Outputs
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents,
            config: {
                systemInstruction,
                responseMimeType: 'application/json',
                responseSchema: quizResponseSchema,
                temperature: 0.4
            }
        });

        const parsedData = JSON.parse(response.text);

        if (!parsedData.questions || parsedData.questions.length === 0) {
            throw new Error('AI returned an empty question set.');
        }

        // Build standardized session object matching the frontend schema
        const newSession = {
            id: `session_${Date.now()}`,
            createdAt: new Date().toISOString(),
            title: parsedData.title || 'Study Review Session',
            subject: parsedData.subject || 'General Studies',
            difficulty,
            summary: parsedData.summary || '',
            keyTakeaways: parsedData.keyTakeaways || [],
            questions: parsedData.questions.map((q) => ({
                question: q.question,
                options: q.options.slice(0, 4),
                correctIndex: Math.min(Math.max(0, Number(q.correctIndex) || 0), 3),
                explanation: q.explanation
            })),
            userAnswers: new Array(parsedData.questions.length).fill(null),
            bestScore: 0,
            completed: false
        };

        // Persist automatically in Revision Vault
        const sessions = await readSessions();
        sessions.unshift(newSession);
        await writeSessions(sessions);

        return res.status(201).json(newSession);
    } catch (error) {
        console.error('Error in /api/generate-quiz:', error);
        return res.status(500).json({
            error: 'Failed to generate study quiz.',
            details: error.message
        });
    }
});

/**
 * GET /api/sessions
 * Retrieve all saved study sessions from the Revision Vault (supports ?q= search).
 */
app.get('/api/sessions', async (req, res) => {
    try {
        const sessions = await readSessions();
        const query = (req.query.q || '').toLowerCase().trim();

        if (!query) {
            return res.json(sessions);
        }

        const filtered = sessions.filter(
            (s) =>
                s.title.toLowerCase().includes(query) ||
                s.subject.toLowerCase().includes(query) ||
                s.summary.toLowerCase().includes(query)
        );
        return res.json(filtered);
    } catch (error) {
        console.error('Error in GET /api/sessions:', error);
        return res.status(500).json({ error: 'Failed to load sessions.' });
    }
});

/**
 * GET /api/sessions/:id
 * Retrieve a single saved study session by ID.
 */
app.get('/api/sessions/:id', async (req, res) => {
    try {
        const sessions = await readSessions();
        const session = sessions.find((s) => s.id === req.params.id);
        if (!session) {
            return res.status(404).json({ error: 'Study session not found.' });
        }
        return res.json(session);
    } catch (error) {
        return res.status(500).json({ error: 'Failed to fetch study session.' });
    }
});

/**
 * PATCH /api/sessions/:id
 * Update user progress, selected answers, and best score for a session.
 */
app.patch('/api/sessions/:id', async (req, res) => {
    try {
        const sessions = await readSessions();
        const idx = sessions.findIndex((s) => s.id === req.params.id);
        if (idx === -1) {
            return res.status(404).json({ error: 'Study session not found.' });
        }

        const session = sessions[idx];
        const { userAnswers, bestScore, completed } = req.body;

        if (Array.isArray(userAnswers)) {
            session.userAnswers = userAnswers;
        }
        if (typeof bestScore === 'number') {
            session.bestScore = Math.max(session.bestScore || 0, bestScore);
        }
        if (typeof completed === 'boolean') {
            session.completed = completed;
        }

        sessions[idx] = session;
        await writeSessions(sessions);

        return res.json(session);
    } catch (error) {
        console.error('Error in PATCH /api/sessions/:id:', error);
        return res.status(500).json({ error: 'Failed to update session progress.' });
    }
});

/**
 * DELETE /api/sessions/:id
 * Delete a study session from the Revision Vault.
 */
app.delete('/api/sessions/:id', async (req, res) => {
    try {
        const sessions = await readSessions();
        const updated = sessions.filter((s) => s.id !== req.params.id);

        if (updated.length === sessions.length) {
            return res.status(404).json({ error: 'Study session not found.' });
        }

        await writeSessions(updated);
        return res.json({ deleted: true, id: req.params.id });
    } catch (error) {
        console.error('Error in DELETE /api/sessions/:id:', error);
        return res.status(500).json({ error: 'Failed to delete session.' });
    }
});

// ============================================================================
// 5. GLOBAL ERROR HANDLER & SERVER STARTUP
// ============================================================================
app.use((err, _req, res, _next) => {
    console.error('Unhandled Server Error:', err.message);
    res.status(400).json({ error: err.message || 'Bad Request' });
});

ensureStorageReady().then(() => {
    app.listen(PORT, () => {
        console.log(`🚀 Padhaku AI Backend running at http://localhost:${3000}`);
    });
});
