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
import Checkbox from '@mui/material/Checkbox';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

import Highlight from 'react-highlight-words';

import { useDocumentExtraction } from './useDocumentExtraction';

interface UploadedFile {
  file_id: string;
  filename: string;
  status: string;
  created_at: string;
}

interface Source {
  filename: string;
  excerpt: string;
  relevance_score: number;
}

type Message = {
  role: 'user' | 'assistant' | 'system';
  content: string;
  sources?: Source[];
};

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
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [showSearchResults, setShowSearchResults] = useState(false);

  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [filesLoading, setFilesLoading] = useState(true);

  // Multi-file selection & active chat state
  const [selectedFileIds, setSelectedFileIds] = useState<Set<string>>(new Set());
  const [activeChatFileIds, setActiveChatFileIds] = useState<string[]>([]);
  const [activeChatFilenames, setActiveChatFilenames] = useState<string[]>([]);

  const API_BASE = import.meta.env.VITE_DWANI_API_BASE_URL || 'https://discovery-server.dwani.ai';
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
    setActiveChatFileIds([]);
    setActiveChatFilenames([]);
  };

  const toggleSearch = () => {
    setShowSearchResults(prev => !prev);
    if (showSearchResults) setSearchQuery('');
  };

  // Open chat with one or more files
  const openChatForFiles = (fileIds: string[], filenames: string[]) => {
    if (fileIds.length === 0) return;

    setActiveChatFileIds(fileIds);
    setActiveChatFilenames(filenames);
    setSelectedFileIds(new Set()); // clear selection

    const count = filenames.length;

    setChatHistory([
      {
        role: 'assistant',
        content: `I've loaded ${count} document${count > 1 ? 's' : ''}: ${filenames.join(', ')}.\n\nAsk me anything — I'll search across all of them!`
      },
    ]);
    setChatOpen(true);
  };

  const handleSendMessage = async () => {
    if (!userMessage.trim() || chatLoading || activeChatFileIds.length === 0) return;

    const userMsg = userMessage.trim();
    setUserMessage('');
    setChatLoading(true);
    setChatError(null);

    const newUserMessage: Message = { role: 'user', content: userMsg };
    setChatHistory(prev => [...prev, newUserMessage]);

    // Temporary placeholder for assistant response
    const tempAssistantMessage: Message = { role: 'assistant', content: '', sources: [] };
    setChatHistory(prev => [...prev, tempAssistantMessage]);

    try {
      const res = await fetch(`${API_BASE}/chat-with-document`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-KEY': API_KEY || '',
        },
        body: JSON.stringify({
          file_ids: activeChatFileIds,
          messages: [
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
      const answer = data.answer?.trim() || 'No reply.';
      const sources: Source[] = data.sources || [];

      // Replace temp message with full answer + sources
      setChatHistory(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: answer,
          sources
        };
        return updated;
      });

    } catch (err) {
      setChatError(err instanceof Error ? err.message : 'Failed to get response');
      // Remove user message and temp assistant on error
      setChatHistory(prev => prev.slice(0, -2));
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

        {/* Multi-select Alert */}
        {selectedFileIds.size > 0 && (
          <Alert
            severity="info"
            action={
              <Button
                color="inherit"
                size="small"
                variant="outlined"
                onClick={() => {
                  const selected = uploadedFiles
                    .filter(f => selectedFileIds.has(f.file_id))
                    .filter(f => f.status === 'completed');
                  if (selected.length > 0) {
                    openChatForFiles(
                      selected.map(f => f.file_id),
                      selected.map(f => f.filename)
                    );
                  }
                }}
              >
                Chat with {selectedFileIds.size} selected
              </Button>
            }
            sx={{ mb: 2 }}
          >
            {selectedFileIds.size} document{selectedFileIds.size > 1 ? 's' : ''} selected
          </Alert>
        )}

        {/* Uploaded Files Table */}
        <Box>
          <Typography variant="h5" gutterBottom sx={{ mb: 2 }}>
            Your Uploaded Documents
          </Typography>
          <TableContainer component={Paper} elevation={2} sx={{ maxHeight: 400, overflow: 'auto', borderRadius: 2 }}>
            <Table stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell padding="checkbox" sx={{ bgcolor: 'background.paper' }} />
                  <TableCell sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Filename</TableCell>
                  <TableCell sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Uploaded</TableCell>
                  <TableCell sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Status</TableCell>
                  <TableCell align="center" sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filesLoading ? (
                  <TableRow>
                    <TableCell colSpan={5} align="center" sx={{ py: 4 }}>
                      <CircularProgress />
                    </TableCell>
                  </TableRow>
                ) : uploadedFiles.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} align="center" sx={{ py: 8, color: 'text.secondary' }}>
                      No documents uploaded yet.
                    </TableCell>
                  </TableRow>
                ) : (
                  uploadedFiles.map(doc => (
                    <TableRow key={doc.file_id} hover selected={selectedFileIds.has(doc.file_id)}>
                      <TableCell padding="checkbox">
                        <Checkbox
                          checked={selectedFileIds.has(doc.file_id)}
                          onChange={(e) => {
                            const newSet = new Set(selectedFileIds);
                            if (e.target.checked) newSet.add(doc.file_id);
                            else newSet.delete(doc.file_id);
                            setSelectedFileIds(newSet);
                          }}
                          disabled={doc.status !== 'completed'}
                        />
                      </TableCell>
                      <TableCell>
                        <Tooltip title={doc.filename}>
                          <Typography noWrap sx={{ maxWidth: 300 }}>
                            {doc.filename}
                          </Typography>
                        </Tooltip>
                      </TableCell>
                      <TableCell>{formatDate(doc.created_at)}</TableCell>
                      <TableCell>
                        <Chip label={getStatusText(doc.status)} color={getStatusColor(doc.status)} size="small" />
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="outlined"
                          size="small"
                          onClick={() => openChatForFiles([doc.file_id], [doc.filename])}
                          disabled={doc.status !== 'completed'}
                        >
                          Chat (Single)
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

        {/* Simplified success message */}
        {status === 'completed' && fileId && (
          <Alert severity="success" sx={{ mt: 2 }}>
            Extraction complete! You can now chat with the document or download the clean PDF.
          </Alert>
        )}

        {status === 'completed' && fileId && (
          <Stack direction="row" spacing={2} sx={{ mt: 2, justifyContent: 'flex-end' }}>
            <Button variant="outlined" onClick={handleOpenPreview}>Preview PDF</Button>
            <Button variant="contained" onClick={handleDownloadPdf}>Download PDF</Button>
          </Stack>
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

      {/* Chat Dialog with Source Citations */}
      <Dialog open={chatOpen} onClose={handleCloseChat} maxWidth="md" fullWidth>
        <DialogTitle>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">
              Chat with {activeChatFilenames.length === 1
                ? activeChatFilenames[0]
                : `${activeChatFilenames.length} documents`}
            </Typography>
            <Button
              startIcon={<SearchIcon />}
              onClick={toggleSearch}
              variant={showSearchResults ? "contained" : "outlined"}
              size="small"
            >
              {showSearchResults ? 'Close Search' : 'Search'}
            </Button>
          </Stack>
        </DialogTitle>

        <DialogContent dividers sx={{ display: 'flex', flexDirection: 'column', height: '70vh' }}>
          {showSearchResults && activeChatFileIds.length === 1 && extractedText && (
            <Paper variant="outlined" sx={{ p: 2, mb: 2, maxHeight: '40vh', overflow: 'auto', bgcolor: 'grey.50' }}>
              <Typography variant="subtitle1" gutterBottom>Search in document</Typography>
              <TextField
                fullWidth variant="outlined" placeholder="Search..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                sx={{ mb: 2 }}
                autoFocus
              />
              {searchQuery.trim() && (
                <Typography component="div" sx={{ whiteSpace: 'pre-wrap', fontSize: '0.95rem', lineHeight: 1.6 }}>
                  <Highlight
                    highlightClassName="search-highlight"
                    searchWords={[searchQuery.trim()]}
                    autoEscape
                    textToHighlight={extractedText}
                  />
                </Typography>
              )}
            </Paper>
          )}

          <Paper variant="outlined" sx={{ flexGrow: 1, overflow: 'auto', p: 2, mb: 2 }}>
            <List>
              {chatHistory.filter(m => m.role !== 'system').map((msg, i) => (
                <ListItem key={i} sx={{ justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                  <Paper
                    elevation={1}
                    sx={{
                      p: 2,
                      maxWidth: '85%',
                      bgcolor: msg.role === 'user' ? 'primary.light' : 'grey.100',
                      color: msg.role === 'user' ? 'white' : 'text.primary'
                    }}
                  >
                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                      {msg.role === 'user' ? 'You' : 'Assistant'}
                    </Typography>

                    <Typography whiteSpace="pre-wrap" sx={{ mb: msg.sources && msg.sources.length > 0 ? 2 : 0 }}>
                      {msg.content || <i>Thinking...</i>}
                    </Typography>

                    {/* Source Citations Accordion */}
                    {msg.sources && msg.sources.length > 0 && (
                      <Accordion elevation={0} sx={{ bgcolor: 'background.paper', mt: 1 }}>
                        <AccordionSummary
                          expandIcon={<ExpandMoreIcon />}
                          sx={{ fontSize: '0.875rem', minHeight: '36px', '& .MuiAccordionSummary-content': { my: 0.5 } }}
                        >
                          <Typography variant="caption" fontWeight="medium">
                            Sources ({msg.sources.length})
                          </Typography>
                        </AccordionSummary>
                        <AccordionDetails sx={{ pt: 0 }}>
                          <Stack spacing={2}>
                            {msg.sources.map((source, idx) => (
                              <Box key={idx} sx={{ fontSize: '0.875rem' }}>
                                <Typography variant="caption" color="text.secondary" gutterBottom>
                                  <strong>{source.filename}</strong> (relevance: {source.relevance_score.toFixed(2)})
                                </Typography>
                                <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'grey.50', mt: 0.5 }}>
                                  <Highlight
                                    highlightClassName="source-highlight"
                                    searchWords={userMessage.split(' ').filter(w => w.length > 3)}
                                    textToHighlight={source.excerpt}
                                    autoEscape={true}
                                  />
                                </Paper>
                              </Box>
                            ))}
                          </Stack>
                        </AccordionDetails>
                      </Accordion>
                    )}
                  </Paper>
                </ListItem>
              ))}
              {chatLoading && (
                <ListItem>
                  <CircularProgress size={24} />
                  <Typography sx={{ ml: 2 }}>Searching documents and thinking...</Typography>
                </ListItem>
              )}
            </List>
          </Paper>

          {chatError && <Alert severity="error" onClose={() => setChatError(null)} sx={{ mb: 2 }}>{chatError}</Alert>}

          <Stack direction="row" spacing={1} alignItems="flex-end">
            <TextField
              fullWidth
              multiline
              maxRows={4}
              variant="outlined"
              placeholder="Ask a question about the document(s)..."
              value={userMessage}
              onChange={e => setUserMessage(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSendMessage())}
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