import { useState } from 'react';
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

import Highlight from 'react-highlight-words';

import { useDocumentExtraction } from './useDocumentExtraction';

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

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearchResults, setShowSearchResults] = useState(false);

  //const API_BASE = import.meta.env.VITE_API_BASE_URL || 'https://discovery-server.dwani.ai';
  const API_BASE = 'http://localhost:8000'
  const API_KEY = import.meta.env.VITE_DWANI_API_KEY;

  const getStatusColor = (status: string | null) => {
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

  const getStatusText = (status: string | null) => {
    switch (status) {
      case 'pending':
        return 'Uploaded – Waiting to process';
      case 'processing':
        return 'Extracting text from PDF...';
      case 'completed':
        return 'Extraction Complete';
      case 'failed':
        return 'Extraction Failed';
      default:
        return 'Ready to upload';
    }
  };

  const handleOpenPreview = () => {
    handlePreviewPdf();
    setPreviewOpen(true);
  };

  const handleClosePreview = () => {
    setPreviewOpen(false);
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

  const handleCloseChat = () => {
    setChatOpen(false);
  };

  const toggleSearch = () => {
    setShowSearchResults(!showSearchResults);
    if (showSearchResults) {
      setSearchQuery('');
    }
  };

  const handleSendMessage = async () => {
    if (!userMessage.trim() || chatLoading || !fileId) return;

    const userMsg = userMessage.trim();
    const updatedHistory = [...chatHistory, { role: 'user', content: userMsg }];
    setChatHistory(updatedHistory);
    setUserMessage('');
    setChatLoading(true);
    setChatError(null);

    try {
      const response = await fetch(`${API_BASE}/chat-with-document`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-KEY': API_KEY || '',
        },
        body: JSON.stringify({
          file_id: fileId,
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
        spacing={4}
        useFlexGap
        sx={{ alignItems: 'center', width: { xs: '100%', sm: '70%' }, maxWidth: '800px' }}
      >
        <Divider sx={{ width: '100%' }} />
        
        <Typography variant="h4" sx={{ textAlign: 'center', fontWeight: 'bold' }}>
          Document Text Extraction
        </Typography>
        
        <Typography sx={{ textAlign: 'center', color: 'text.secondary' }}>
          Upload a PDF document and we'll extract clean, readable plain text using advanced vision models.
        </Typography>

        {/* Upload Controls */}
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ alignItems: 'center', width: '100%' }}>
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            style={{ display: 'none' }}
            id="pdf-upload"
            disabled={!!fileId}
          />
          <label htmlFor="pdf-upload">
            <Button
              variant="outlined"
              component="span"
              disabled={!!fileId}
            >
              {file ? 'Change PDF' : 'Upload PDF'}
            </Button>
          </label>

          <Button
            variant="contained"
            color="primary"
            onClick={handleStartExtraction}
            disabled={!file || !!fileId || loading}
            startIcon={uploadLoading ? <CircularProgress size={20} color="inherit" /> : null}
            sx={{ px: 4, py: 1.5 }}
          >
            {uploadLoading ? 'Uploading...' : fileId ? 'Processing...' : 'Start Extraction'}
          </Button>

          {fileId && status && (
            <Button variant="text" onClick={reset}>
              Reset
            </Button>
          )}
        </Stack>

        {/* File & Status Info */}
        {file && (
          <Typography sx={{ color: 'text.secondary' }}>
            Selected: <strong>{file.name}</strong>
          </Typography>
        )}

        {fileId && (
          <Chip
            label={getStatusText(status)}
            color={getStatusColor(status)}
            icon={loading ? <CircularProgress size={16} color="inherit" /> : undefined}
            sx={{ fontWeight: 'medium', px: 1 }}
          />
        )}

        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ width: '100%' }} onClose={clearError}>
            {error}
          </Alert>
        )}

        {/* Extracted Text Result */}
        {extractedText && status === 'completed' && (
          <Box
            sx={{
              mt: 3,
              p: 3,
              bgcolor: 'background.paper',
              borderRadius: 2,
              border: '1px solid',
              borderColor: 'divider',
              width: '100%',
              boxShadow: 1,
            }}
          >
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'medium' }}>
              Extracted Text
            </Typography>
            <Typography
              component="pre"
              sx={{
                mt: 2,
                color: 'text.primary',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                fontFamily: 'inherit',
                lineHeight: 1.6,
                maxHeight: '60vh',
                overflow: 'auto',
                bgcolor: 'grey.50',
                p: 2,
                borderRadius: 1,
              }}
            >
              {extractedText}
            </Typography>

            <Stack direction="row" spacing={2} sx={{ mt: 3, justifyContent: 'flex-end' }}>
              <Button
                variant="outlined"
                onClick={() => {
                  navigator.clipboard.writeText(extractedText);
                }}
              >
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

      {/* PDF Preview Dialog */}
      <Dialog open={previewOpen} onClose={handleClosePreview} maxWidth="lg" fullWidth>
        <DialogTitle>Regenerated PDF Preview</DialogTitle>
        <DialogContent>
          {previewUrl ? (
            <iframe
              src={previewUrl}
              style={{ width: '100%', height: '70vh', border: 'none' }}
              title="PDF Preview"
            />
          ) : (
            <CircularProgress sx={{ display: 'block', mx: 'auto' }} />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClosePreview}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Chat with Document Dialog + Search */}
      <Dialog open={chatOpen} onClose={handleCloseChat} maxWidth="md" fullWidth>
        <DialogTitle>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">Chat with Your Document</Typography>
            <Button
              startIcon={<SearchIcon />}
              onClick={toggleSearch}
              variant={showSearchResults ? "contained" : "outlined"}
              color="primary"
              size="small"
            >
              {showSearchResults ? 'Close Search' : 'Search in Document'}
            </Button>
          </Stack>
        </DialogTitle>

        <DialogContent dividers sx={{ display: 'flex', flexDirection: 'column', height: '70vh' }}>
          {/* Search Results Panel */}
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
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  ),
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
                <Typography
                  component="div"
                  sx={{
                    whiteSpace: 'pre-wrap',
                    fontSize: '0.95rem',
                    lineHeight: 1.6,
                  }}
                >
                  <Highlight
                    highlightClassName="search-highlight"
                    searchWords={[searchQuery.trim()]}
                    autoEscape={true}
                    textToHighlight={extractedText || ''}
                  />
                </Typography>
              ) : (
                <Typography color="text.secondary">
                  Type a keyword or phrase to search within the document.
                </Typography>
              )}
            </Paper>
          )}

          {/* Chat History */}
          <Paper variant="outlined" sx={{ flexGrow: 1, overflow: 'auto', p: 2, mb: 2, bgcolor: 'background.default' }}>
            <List>
              {chatHistory
                .filter(msg => msg.role !== 'system')
                .map((msg, idx) => (
                  <ListItem
                    key={idx}
                    alignItems="flex-start"
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

          {/* Chat Input */}
          <Stack direction="row" spacing={1} alignItems="flex-end">
            <TextField
              fullWidth
              multiline
              maxRows={4}
              variant="outlined"
              placeholder="Ask a question about the document..."
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
          <Button onClick={handleCloseChat}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}