import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Chip from '@mui/material/Chip';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import TextField from '@mui/material/TextField';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import Paper from '@mui/material/Paper';
import SendIcon from '@mui/icons-material/Send';
import SearchIcon from '@mui/icons-material/Search';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Tooltip from '@mui/material/Tooltip';

import Highlight from 'react-highlight-words';

import { useDocumentExtraction } from './useDocumentExtraction';

interface UploadedFile {
  file_id: string;
  filename: string;
  status: string;
  created_at: string;
}

type Message = { role: 'user' | 'assistant' | 'system'; content: string };

export default function Digitiser() {
  const {
    file,
    fileId,
    extractedText,
    status,
    loading,
    uploadLoading,
    error,
    previewUrl,
    handleFileChange,
    handleStartExtraction,
    handleDownloadPdf,
    handlePreviewPdf,
    loadExistingFile,
    reset,
    clearError
  } = useDocumentExtraction();

  const [previewOpen, setPreviewOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [userMessage, setUserMessage] = useState('');
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [currentFilename, setCurrentFilename] = useState<string>('');

  const [searchQuery, setSearchQuery] = useState('');
  const [showSearchResults, setShowSearchResults] = useState(false);

  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [filesLoading, setFilesLoading] = useState(true);

  const API_BASE = import.meta.env.VITE_DWANI_API_BASE_URL || 'https://discovery-server.dwani.ai';
  //const API_BASE = 'http://localhost:8000'
  const API_KEY = import.meta.env.VITE_DWANI_API_KEY;

  const fetchUploadedFiles = async () => {
    setFilesLoading(true);
    try {
      const response = await fetch(`${API_BASE}/files/`, {
        headers: { 'X-API-KEY': API_KEY || '' },
      });
      if (response.ok) {
        const data = await response.json();
        setUploadedFiles(data);
      }
    } catch (err) {
      console.error('Failed to fetch files list');
    } finally {
      setFilesLoading(false);
    }
  };

  useEffect(() => {
    fetchUploadedFiles();
  }, []);

  useEffect(() => {
    if (fileId && status === 'completed') {
      fetchUploadedFiles();
    }
  }, [fileId, status]);

  const getStatusColor = (s: string) => {
    switch (s) {
      case 'pending':
      case 'processing':
        return 'warning';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusText = (s: string) => {
    switch (s) {
      case 'pending': return 'Waiting';
      case 'processing': return 'Processing';
      case 'completed': return 'Ready';
      case 'failed': return 'Failed';
      default: return 'Unknown';
    }
  };

  const formatDate = (dateStr: string) => new Date(dateStr).toLocaleString();

  const handleOpenPreview = () => {
    handlePreviewPdf();
    setPreviewOpen(true);
  };

  const handleClosePreview = () => setPreviewOpen(false);

  const handleCloseChat = () => {
    setChatOpen(false);
    setChatHistory([]);
    setSearchQuery('');
    setShowSearchResults(false);
  };

  const toggleSearch = () => {
    setShowSearchResults(prev => !prev);
    if (showSearchResults) setSearchQuery('');
  };

  const openChatForFile = async (fileId: string, filename: string) => {
  try {
    await loadExistingFile(fileId);
    if (status === 'completed') {
      setCurrentFilename(filename);
      setChatHistory([
        { role: 'assistant' as const, content: `I've loaded "${filename}". Ask me anything!` },
      ]);
      setChatOpen(true);
    } else {
      alert('Document is not ready yet. Please wait for extraction to complete.');
    }
  } catch {
    alert('Failed to load document.');
  }
};
  

  const handleOpenChat = () => {
  if (!extractedText || !fileId) return;
  setCurrentFilename(file?.name || 'Document');

  // Clear any old system messages and start fresh
  setChatHistory([
    { role: 'assistant' as const, content: 'Hello! I’ve loaded the document. Ask me anything about it.' },
  ]);
  setChatOpen(true);
};

const handleSendMessage = async () => {
  if (!userMessage.trim() || chatLoading || !fileId) return;

  const userMsg = userMessage.trim();
  setUserMessage('');
  setChatLoading(true);
  setChatError(null);

  // Append user message immediately for UI responsiveness
  const newUserMessage: Message = { role: 'user', content: userMsg };
  setChatHistory(prev => [...prev, newUserMessage]);

  try {
    const res = await fetch(`${API_BASE}/chat-with-document`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': API_KEY || '',
      },
      body: JSON.stringify({
        file_id: fileId,
        messages: [
          // Only send visible conversation history (no hidden system prompt)
          ...chatHistory.filter(m => m.role !== 'system'),
          newUserMessage
        ]
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Chat failed');
    }

    const data = await res.json();
    const answer = data.answer?.trim() || 'No response.';

    // Optional: show how many sources were used
    const sourcesInfo = data.sources !== undefined ? ` (${data.sources} source${data.sources === 1 ? '' : 's'})` : '';

    setChatHistory(prev => [
      ...prev,
      { role: 'assistant' as const, content: answer + sourcesInfo }
    ]);
  } catch (err) {
    setChatError(err instanceof Error ? err.message : 'Failed to get response');
    // Remove the pending user message on error
    setChatHistory(prev => prev.filter(m => m !== newUserMessage));
  } finally {
    setChatLoading(false);
  }
};


  return (
    <Box sx={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      p: { xs: 4, sm: 6 },
      pt: { xs: 10, sm: 12 },
      minHeight: '100vh',
      bgcolor: 'background.default',
    }}>
      <Stack spacing={5} sx={{ width: { xs: '100%', sm: '90%', md: '1000px' } }}>

        {/* Uploaded Files Table - Now Scrollable */}
        <Box>
          <Typography variant="h5" gutterBottom sx={{ mb: 2 }}>
            Your Uploaded Documents
          </Typography>
          <TableContainer 
            component={Paper} 
            elevation={2}
            sx={{ 
              maxHeight: 400, 
              overflow: 'auto',
              borderRadius: 2,
            }}
          >
            <Table stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Filename</TableCell>
                  <TableCell sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Uploaded</TableCell>
                  <TableCell sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Status</TableCell>
                  <TableCell align="center" sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filesLoading ? (
                  <TableRow>
                    <TableCell colSpan={4} align="center" sx={{ py: 4 }}>
                      <CircularProgress />
                    </TableCell>
                  </TableRow>
                ) : uploadedFiles.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={4} align="center" sx={{ py: 8, color: 'text.secondary' }}>
                      No documents uploaded yet.
                    </TableCell>
                  </TableRow>
                ) : (
                  uploadedFiles.map(doc => (
                    <TableRow key={doc.file_id} hover>
                      <TableCell>
                        <Tooltip title={doc.filename}>
                          <Typography noWrap sx={{ maxWidth: 300 }}>
                            {doc.filename}
                          </Typography>
                        </Tooltip>
                      </TableCell>
                      <TableCell>{formatDate(doc.created_at)}</TableCell>
                      <TableCell>
                        <Chip 
                          label={getStatusText(doc.status)} 
                          color={getStatusColor(doc.status)} 
                          size="small" 
                        />
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="outlined"
                          size="small"
                          onClick={() => openChatForFile(doc.file_id, doc.filename)}
                          disabled={doc.status !== 'completed'}
                        >
                          Chat
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
        <Divider />

        {/* New Upload Section */}
        <Typography variant="h5" gutterBottom>Upload New Document</Typography>

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ alignItems: 'center' }}>
          <input type="file" accept="application/pdf" onChange={handleFileChange} style={{ display: 'none' }} id="pdf-upload" disabled={!!fileId} />
          <label htmlFor="pdf-upload">
            <Button variant="outlined" component="span" disabled={!!fileId}>
              {file ? 'Change PDF' : 'Upload PDF'}
            </Button>
          </label>

          <Button
            variant="contained"
            onClick={handleStartExtraction}
            disabled={!file || !!fileId || loading}
            startIcon={uploadLoading ? <CircularProgress size={20} color="inherit" /> : null}
          >
            {uploadLoading ? 'Uploading...' : fileId ? 'Processing...' : 'Start Extraction'}
          </Button>

          {fileId && <Button variant="text" onClick={reset}>Reset</Button>}
        </Stack>

        {file && <Typography sx={{ color: 'text.secondary' }}>Selected: <strong>{file.name}</strong></Typography>}

        {fileId && (
          <Chip
            label={getStatusText(status || '')}
            color={getStatusColor(status || '')}
            icon={loading ? <CircularProgress size={16} color="inherit" /> : undefined}
          />
        )}

        {error && <Alert severity="error" onClose={clearError} sx={{ width: '100%' }}>{error}</Alert>}

        {/* Current Document Result */}
        {extractedText && status === 'completed' && (
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 2, border: '1px solid', borderColor: 'divider', boxShadow: 1 }}>
            <Typography variant="h6" gutterBottom>Extracted Text ({file?.name || currentFilename})</Typography>
            <Typography component="pre" sx={{
              whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: '60vh', overflow: 'auto',
              bgcolor: 'grey.50', p: 2, borderRadius: 1, lineHeight: 1.6
            }}>
              {extractedText}
            </Typography>

            <Stack direction="row" spacing={2} sx={{ mt: 3, justifyContent: 'flex-end' }}>
              <Button variant="outlined" onClick={() => navigator.clipboard.writeText(extractedText)}>Copy Text</Button>
              <Button variant="outlined" onClick={handleOpenPreview}>Preview PDF</Button>
              <Button variant="contained" onClick={handleDownloadPdf}>Download PDF</Button>
              <Button variant="outlined" color="secondary" onClick={handleOpenChat}>Chat with Document</Button>
            </Stack>
          </Box>
        )}
      </Stack>

      {/* Preview Dialog */}
      <Dialog open={previewOpen} onClose={handleClosePreview} maxWidth="lg" fullWidth>
        <DialogTitle>Regenerated PDF Preview</DialogTitle>
        <DialogContent>
          {previewUrl ? <iframe src={previewUrl} style={{ width: '100%', height: '80vh', border: 'none' }} /> : <CircularProgress sx={{ display: 'block', mx: 'auto', my: 4 }} />}
        </DialogContent>
        <DialogActions><Button onClick={handleClosePreview}>Close</Button></DialogActions>
      </Dialog>

      {/* Chat Dialog with Search */}
      <Dialog open={chatOpen} onClose={handleCloseChat} maxWidth="md" fullWidth>
        <DialogTitle>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">Chat with {currentFilename || 'Document'}</Typography>
            <Button startIcon={<SearchIcon />} onClick={toggleSearch} variant={showSearchResults ? "contained" : "outlined"} size="small">
              {showSearchResults ? 'Close Search' : 'Search'}
            </Button>
          </Stack>
        </DialogTitle>

        <DialogContent dividers sx={{ display: 'flex', flexDirection: 'column', height: '70vh' }}>
          {showSearchResults && (
            <Paper variant="outlined" sx={{ p: 2, mb: 2, maxHeight: '40vh', overflow: 'auto', bgcolor: 'grey.50' }}>
              <Typography variant="subtitle1" gutterBottom>Search "{searchQuery || '—'}"</Typography>
              <TextField
                fullWidth variant="outlined" placeholder="Search in document..."
                value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                sx={{ mb: 2 }} autoFocus
              />
              {searchQuery.trim() && (
                <Typography component="div" sx={{ whiteSpace: 'pre-wrap', fontSize: '0.95rem', lineHeight: 1.6 }}>
                  <Highlight 
                    highlightClassName="search-highlight" 
                    searchWords={[searchQuery.trim()]} 
                    autoEscape 
                    textToHighlight={extractedText ?? ''} 
                  />
                </Typography>
              )}
            </Paper>
          )}

          <Paper variant="outlined" sx={{ flexGrow: 1, overflow: 'auto', p: 2, mb: 2 }}>
            <List>
              {chatHistory.filter(m => m.role !== 'system').map((msg, i) => (
                <ListItem key={i} sx={{ justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                  <Paper elevation={1} sx={{
                    p: 2, maxWidth: '80%',
                    bgcolor: msg.role === 'user' ? 'primary.light' : 'grey.100',
                    color: msg.role === 'user' ? 'white' : 'text.primary'
                  }}>
                    <Typography variant="subtitle2" fontWeight="bold">{msg.role === 'user' ? 'You' : 'Assistant'}</Typography>
                    <Typography whiteSpace="pre-wrap">{msg.content}</Typography>
                  </Paper>
                </ListItem>
              ))}
              {chatLoading && (
                <ListItem>
                  <CircularProgress size={24} />
                  <Typography sx={{ ml: 2 }}>Searching document and thinking...</Typography>
                </ListItem>
              )}            
            </List>
          </Paper>

          {chatError && <Alert severity="error" onClose={() => setChatError(null)} sx={{ mb: 2 }}>{chatError}</Alert>}

          <Stack direction="row" spacing={1} alignItems="flex-end">
            <TextField
              fullWidth multiline maxRows={4} variant="outlined"
              placeholder="Ask a question..."
              value={userMessage} onChange={e => setUserMessage(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSendMessage())}
              disabled={chatLoading}
            />
            <Button variant="contained" onClick={handleSendMessage} disabled={!userMessage.trim() || chatLoading} sx={{ height: 56 }}>
              <SendIcon />
            </Button>
          </Stack>
        </DialogContent>

        <DialogActions><Button onClick={handleCloseChat}>Close</Button></DialogActions>
      </Dialog>
    </Box>
  );
}