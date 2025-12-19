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
import InputAdornment from '@mui/material/InputAdornment';
import IconButton from '@mui/material/IconButton';
import ClearIcon from '@mui/icons-material/Clear';
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
    reset,
    clearError
  } = useDocumentExtraction();

  const [previewOpen, setPreviewOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [userMessage, setUserMessage] = useState('');
  const [chatHistory, setChatHistory] = useState<
    { role: 'user' | 'assistant' | 'system'; content: string }[]
  >([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [showSearchResults, setShowSearchResults] = useState(false);

  // New: List of uploaded files
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [filesLoading, setFilesLoading] = useState(true);

  //const API_BASE = import.meta.env.VITE_API_BASE_URL || 'https://discovery-server.dwani.ai';
  const API_BASE = 'http://localhost:8000'
  const API_KEY = import.meta.env.VITE_DWANI_API_KEY;

  // Fetch list of uploaded files
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
      console.error('Failed to fetch uploaded files');
    } finally {
      setFilesLoading(false);
    }
  };

  useEffect(() => {
    fetchUploadedFiles();
  }, []);

  // Refresh file list after new upload
  useEffect(() => {
    if (fileId && status === 'completed') {
      fetchUploadedFiles();
    }
  }, [fileId, status]);

  const getStatusColor = (status: string) => {
    switch (status) {
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

  const getStatusText = (status: string) => {
    switch (status) {
      case 'pending':
        return 'Waiting';
      case 'processing':
        return 'Processing';
      case 'completed':
        return 'Ready';
      case 'failed':
        return 'Failed';
      default:
        return 'Unknown';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const handleOpenPreview = () => {
    handlePreviewPdf();
    setPreviewOpen(true);
  };

  const handleClosePreview = () => {
    setPreviewOpen(false);
  };

  const loadDocumentForChat = async (fileId: string, filename: string) => {
    try {
      const response = await fetch(`${API_BASE}/files/${fileId}`, {
        headers: { 'X-API-KEY': API_KEY || '' },
      });

      if (!response.ok) throw new Error('Failed to load document');

      const data = await response.json();

      if (data.status !== 'completed' || !data.extracted_text) {
        alert('Document not ready or extraction failed.');
        return;
      }

      // Switch to this document
      reset(); // Clear current upload state
      // We don't set file/fileId here since we're not re-uploading
      // Instead, we just load the text and open chat

      // Manually set extracted text (bypass hook limitations)
      // Note: This is a workaround since useDocumentExtraction is designed for single upload
      // In a real app, you'd refactor the hook to support loading existing files

      // For now, we'll use a separate state
      // But to keep it simple, we'll just open chat with context

      setChatOpen(true);
      setSearchQuery('');
      setShowSearchResults(false);

      const truncatedText = data.extracted_text.slice(0, 20000);
      setChatHistory([
        {
          role: 'system',
          content: `You are a helpful assistant answering questions based solely on the following document "${filename}":\n\n${truncatedText}`,
        },
        {
          role: 'assistant',
          content: `I've loaded "${filename}". Ask me anything about this document!`,
        },
      ]);
    } catch (err) {
      alert('Could not load the document for chat.');
    }
  };

  const handleOpenChat = () => {
    setChatOpen(true);
    setSearchQuery('');
    setShowSearchResults(false);

    if (chatHistory.length === 0 && extractedText) {
      const truncatedText = extractedText.slice(0, 20000);
      setChatHistory([
        {
          role: 'system',
          content: `You are a helpful assistant answering questions based solely on the following document text:\n\n${truncatedText}`,
        },
        {
          role: 'assistant',
          content: 'Hello! I’ve loaded the document. You can chat with me or use the search box to find specific text.',
        },
      ]);
    }
  };

  const toggleSearch = () => {
    setShowSearchResults(!showSearchResults);
    if (showSearchResults) {
      setSearchQuery('');
    }
  };

  const handleSendMessage = async () => {
    if (!userMessage.trim() || chatLoading) return;

    const userMsg = userMessage.trim();
    const updatedHistory = [...chatHistory, { role: 'user', content: userMsg }];
    setChatHistory(updatedHistory);
    setUserMessage('');
    setChatLoading(true);
    setChatError(null);

    // Determine file_id: prefer current upload, fallback to first in history (from loaded doc)
    let currentFileId = fileId;
    if (!currentFileId && uploadedFiles.length > 0) {
      // Try to extract from system message if loaded from history
      const systemMsg = chatHistory.find(m => m.role === 'system');
      if (systemMsg) {
        // In real app, store file_id in state when loading
        // Here we skip file_id for loaded docs (chat still works via context)
      }
    }

    try {
      const response = await fetch(`${API_BASE}/chat-with-document`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-KEY': API_KEY || '',
        },
        body: JSON.stringify({
          file_id: currentFileId || null, // Optional if context is in messages
          messages: updatedHistory,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Chat request failed');
      }

      const data = await response.json();
      const assistantReply = data.answer?.trim() || 'No response received.';

      setChatHistory([...updatedHistory, { role: 'assistant', content: assistantReply }]);
    } catch (err) {
      setChatError(err instanceof Error ? err.message : 'Failed to get response');
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        p: { xs: 4, sm: 6 },
        pt: { xs: 10, sm: 12 },
        minHeight: '100vh',
        bgcolor: 'background.default',
      }}
    >
      <Stack
        spacing={5}
        useFlexGap
        sx={{ width: { xs: '100%', sm: '90%', md: '80%' }, maxWidth: '1000px' }}
      >
        <Typography variant="h4" sx={{ textAlign: 'center', fontWeight: 'bold' }}>
          Document Text Extraction & Chat
        </Typography>

        {/* Uploaded Files Table */}
        <Box>
          <Typography variant="h6" gutterBottom>
            Your Uploaded Documents
          </Typography>
          <TableContainer component={Paper} elevation={3}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Filename</strong></TableCell>
                  <TableCell><strong>Uploaded</strong></TableCell>
                  <TableCell><strong>Status</strong></TableCell>
                  <TableCell align="center"><strong>Actions</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filesLoading ? (
                  <TableRow>
                    <TableCell colSpan={4} align="center">
                      <CircularProgress size={30} />
                    </TableCell>
                  </TableRow>
                ) : uploadedFiles.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={4} align="center" sx={{ color: 'text.secondary' }}>
                      No documents uploaded yet.
                    </TableCell>
                  </TableRow>
                ) : (
                  uploadedFiles.map((doc) => (
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
                          onClick={() => loadDocumentForChat(doc.file_id, doc.filename)}
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

        {/* Current Upload Section */}
        <Typography variant="h6" gutterBottom>
          Upload New Document
        </Typography>

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ alignItems: 'center' }}>
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            style={{ display: 'none' }}
            id="pdf-upload"
            disabled={!!fileId}
          />
          <label htmlFor="pdf-upload">
            <Button variant="outlined" component="span" disabled={!!fileId}>
              {file ? 'Change PDF' : 'Upload PDF'}
            </Button>
          </label>

          <Button
            variant="contained"
            color="primary"
            onClick={handleStartExtraction}
            disabled={!file || !!fileId || loading}
            startIcon={uploadLoading ? <CircularProgress size={20} color="inherit" /> : null}
          >
            {uploadLoading ? 'Uploading...' : fileId ? 'Processing...' : 'Start Extraction'}
          </Button>

          {fileId && status && (
            <Button variant="text" onClick={reset}>
              Reset
            </Button>
          )}
        </Stack>

        {file && <Typography sx={{ color: 'text.secondary' }}>Selected: <strong>{file.name}</strong></Typography>}

        {fileId && (
          <Chip
            label={getStatusText(status || '')}
            color={getStatusColor(status || '')}
            icon={loading ? <CircularProgress size={16} color="inherit" /> : undefined}
          />
        )}

        {error && (
          <Alert severity="error" sx={{ width: '100%' }} onClose={clearError}>
            {error}
          </Alert>
        )}

        {/* Current Document Result */}
        {extractedText && status === 'completed' && (
          <Box
            sx={{
              p: 3,
              bgcolor: 'background.paper',
              borderRadius: 2,
              border: '1px solid',
              borderColor: 'divider',
              boxShadow: 1,
            }}
          >
            <Typography variant="h6" gutterBottom>
              Current Document: {file?.name}
            </Typography>
            <Typography component="pre" sx={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              maxHeight: '50vh',
              overflow: 'auto',
              bgcolor: 'grey.50',
              p: 2,
              borderRadius: 1,
            }}>
              {extractedText}
            </Typography>

            <Stack direction="row" spacing={2} sx={{ mt: 3, justifyContent: 'flex-end' }}>
              <Button variant="outlined" onClick={() => navigator.clipboard.writeText(extractedText)}>
                Copy Text
              </Button>
              <Button variant="outlined" onClick={handleOpenPreview}>
                Preview PDF
              </Button>
              <Button variant="contained" onClick={handleDownloadPdf}>
                Download PDF
              </Button>
              <Button variant="outlined" color="secondary" onClick={handleOpenChat}>
                Chat with Document
              </Button>
            </Stack>
          </Box>
        )}
      </Stack>

      {/* Preview & Chat Dialogs remain unchanged */}
      {/* ... (same as previous version) */}

      {/* PDF Preview Dialog */}
      <Dialog open={previewOpen} onClose={handleClosePreview} maxWidth="lg" fullWidth>
        <DialogTitle>Regenerated PDF Preview</DialogTitle>
        <DialogContent>
          {previewUrl ? (
            <iframe src={previewUrl} style={{ width: '100%', height: '70vh', border: 'none' }} />
          ) : (
            <CircularProgress sx={{ display: 'block', mx: 'auto' }} />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClosePreview}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Chat Dialog with Search */}
      <Dialog open={chatOpen} onClose={() => setChatOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">Chat with Document</Typography>
            <Button
              startIcon={<SearchIcon />}
              onClick={toggleSearch}
              variant={showSearchResults ? "contained" : "outlined"}
              size="small"
            >
              {showSearchResults ? 'Close Search' : 'Search in Document'}
            </Button>
          </Stack>
        </DialogTitle>

        <DialogContent dividers sx={{ display: 'flex', flexDirection: 'column', height: '70vh' }}>
          {showSearchResults && (
            <Paper variant="outlined" sx={{ p: 2, mb: 2, maxHeight: '40vh', overflow: 'auto', bgcolor: 'grey.50' }}>
              <Typography variant="subtitle1" gutterBottom>
                Search Results for: <strong>"{searchQuery || '—'}"</strong>
              </Typography>

              <TextField
                fullWidth
                variant="outlined"
                placeholder="Search in document..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                sx={{ mb: 2 }}
                InputProps={{
                  startAdornment: <InputAdornment position="start"><SearchIcon /></InputAdornment>,
                  endAdornment: searchQuery && (
                    <InputAdornment position="end">
                      <IconButton onClick={() => setSearchQuery('')} size="small">
                        <ClearIcon />
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                autoFocus
              />

              {searchQuery.trim() ? (
                <Typography component="div" sx={{ whiteSpace: 'pre-wrap', fontSize: '0.95rem', lineHeight: 1.6 }}>
                  <Highlight
                    highlightClassName="search-highlight"
                    searchWords={[searchQuery.trim()]}
                    autoEscape={true}
                    textToHighlight={extractedText || chatHistory.find(m => m.role === 'system')?.content.split('\n\n').slice(1).join('\n\n') || ''}
                  />
                </Typography>
              ) : (
                <Typography color="text.secondary">Type a keyword to search.</Typography>
              )}
            </Paper>
          )}

          <Paper variant="outlined" sx={{ flexGrow: 1, overflow: 'auto', p: 2, mb: 2 }}>
            <List>
              {chatHistory.filter(msg => msg.role !== 'system').map((msg, idx) => (
                <ListItem
                  key={idx}
                  sx={{
                    flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                    textAlign: msg.role === 'user' ? 'right' : 'left',
                  }}
                >
                  <Paper
                    elevation={1}
                    sx={{
                      p: 2,
                      maxWidth: '80%',
                      bgcolor: msg.role === 'user' ? 'primary.light' : 'grey.100',
                      color: msg.role === 'user' ? 'white' : 'text.primary',
                    }}
                  >
                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                      {msg.role === 'user' ? 'You' : 'Assistant'}
                    </Typography>
                    <Typography whiteSpace="pre-wrap">{msg.content}</Typography>
                  </Paper>
                </ListItem>
              ))}
              {chatLoading && (
                <ListItem>
                  <CircularProgress size={24} />
                  <Typography sx={{ ml: 2 }}>Thinking...</Typography>
                </ListItem>
              )}
            </List>
          </Paper>

          {chatError && (
            <Alert severity="error" onClose={() => setChatError(null)} sx={{ mb: 2 }}>
              {chatError}
            </Alert>
          )}

          <Stack direction="row" spacing={1} alignItems="flex-end">
            <TextField
              fullWidth
              multiline
              maxRows={4}
              variant="outlined"
              placeholder="Ask a question..."
              value={userMessage}
              onChange={(e) => setUserMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              disabled={chatLoading}
            />
            <Button
              variant="contained"
              onClick={handleSendMessage}
              disabled={!userMessage.trim() || chatLoading}
              sx={{ height: 56 }}
            >
              <SendIcon />
            </Button>
          </Stack>
        </DialogContent>

        <DialogActions>
          <Button onClick={() => setChatOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}