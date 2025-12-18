import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import MainContent from './components/MainContent';
import Latest from './components/Latest';
import AppTheme from '../shared-theme/AppTheme';
import Grid from '@mui/material/Grid2';

export default function Blog(props: { disableCustomTheme?: boolean }) {
  return (
    <AppTheme {...props}>
      <CssBaseline enableColorScheme />
      <Container
        maxWidth="lg"
        component="main"
        sx={{ display: 'flex', flexDirection: 'column', my: 16, gap: 4 }}
      >
        
        <div style={{ display: 'none' }}>
        <MainContent />
        <Latest />
        </div>
      <Grid container spacing={2}>
        <Grid size={{ xs: 12 }} >
        <iframe
          width="560"
          height="315"
          src="https://www.youtube.com/embed/VqFdZAkR_a0?rel=0"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          title="Watch the video"
        ></iframe>
        </Grid>
      </Grid>
      </Container>
    </AppTheme>
  );
}
