package main

import (
    "bytes"
    "context"
    "encoding/base64"
    "encoding/json"
    "errors"
    "fmt"
    "image/jpeg"
    "io"
    "log"
    "net/http"
    "os"
    "regexp"
    "sort"
    "strconv"
    "strings"
    "sync"
    "time"

    "github.com/gen2brain/go-fitz"
    "github.com/gin-gonic/gin"
)

var (
    logger = log.New(os.Stdout, "INFO: ", log.Ldate|log.Ltime|log.Lshortfile)
    re     = regexp.MustCompile(`\x60{3}(?:json)?\s*([\s\S]*?)\s*\x60{3}`)
)

type OpenAIClient struct {
    baseURL string
}

func getOpenAIClient(model string) *OpenAIClient {
    validModels := []string{"gemma3", "gpt-oss"}
    found := false
    for _, m := range validModels {
        if m == model {
            found = true
            break
        }
    }
    if !found {
        panic(fmt.Sprintf("Invalid model: %s", model))
    }

    modelPorts := map[string]string{
        "gemma3": "9000",
        "gpt-oss": "9500",
    }
    vlmBaseURL := os.Getenv("VLLM_IP", "0.0.0.0")
    baseURL := fmt.Sprintf("http://%s:%s/v1", vlmBaseURL, modelPorts[model])
    return &OpenAIClient{baseURL: baseURL}
}

func encodeImage(img image.Image) (string, error) {
    buf := new(bytes.Buffer)
    err := jpeg.Encode(buf, img, &jpeg.Options{Quality: 85})
    if err != nil {
        return "", err
    }
    return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

func cleanResponse(rawResponse string) string {
    if rawResponse == "" {
        return ""
    }
    cleaned := re.ReplaceAllString(rawResponse, "$1")
    return strings.TrimSpace(cleaned)
}

type BatchResult struct {
    Data          map[string]string
    SkippedPages  []int
    Err           error
}

func processSingleBatch(client *OpenAIClient, model string, batchMessages []interface{}, batchStart, batchEnd int) *BatchResult {
    payload := map[string]interface{}{
        "model":      model,
        "messages": []map[string]interface{}{
            {"role": "user", "content": batchMessages},
        },
        "temperature": 0.2,
        "max_tokens":  29695,
    }
    jsonData, err := json.Marshal(payload)
    if err != nil {
        return &BatchResult{Err: err}
    }

    req, err := http.NewRequest("POST", client.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
    if err != nil {
        return &BatchResult{Err: err}
    }
    req.Header.Set("Content-Type", "application/json")

    httpClient := &http.Client{}
    resp, err := httpClient.Do(req)
    if err != nil {
        return &BatchResult{Err: err}
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return &BatchResult{Err: err}
    }

    var response struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
    }
    err = json.Unmarshal(body, &response)
    if err != nil {
        return &BatchResult{Err: err}
    }
    if len(response.Choices) == 0 {
        return &BatchResult{Err: errors.New("no choices in response")}
    }

    rawResponse := response.Choices[0].Message.Content
    cleanedResponse := cleanResponse(rawResponse)
    if cleanedResponse == "" {
        return &BatchResult{SkippedPages: rangeInts(batchStart, batchEnd), Err: errors.New("empty response")}
    }

    var batchResults map[string]string
    err = json.Unmarshal([]byte(cleanedResponse), &batchResults)
    if err != nil {
        return &BatchResult{SkippedPages: rangeInts(batchStart, batchEnd), Err: err}
    }

    return &BatchResult{Data: batchResults}
}

type PageResult struct {
    Data map[string]string
    Err  error
    Page int
}

func processSinglePage(client *OpenAIClient, model string, img image.Image, pageIdx int) *PageResult {
    b64, err := encodeImage(img)
    if err != nil {
        return &PageResult{Page: pageIdx, Err: err}
    }

    singleMessage := []interface{}{
        map[string]interface{}{
            "type":      "image_url",
            "image_url": map[string]string{"url": "data:image/jpeg;base64," + b64},
        },
        map[string]interface{}{
            "type": "text",
            "text": fmt.Sprintf("Extract plain text from this single PDF page (page number %d). Return the result as a valid JSON object where the key is the page number (%d) and the value is the extracted text. Ensure the response is strictly JSON-formatted and does not include markdown code blocks.", pageIdx, pageIdx),
        },
    }

    payload := map[string]interface{}{
        "model":      model,
        "messages": []map[string]interface{}{
            {"role": "user", "content": singleMessage},
        },
        "temperature": 0.2,
        "max_tokens":  29695,
    }
    jsonData, err := json.Marshal(payload)
    if err != nil {
        return &PageResult{Page: pageIdx, Err: err}
    }

    req, err := http.NewRequest("POST", client.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
    if err != nil {
        return &PageResult{Page: pageIdx, Err: err}
    }
    req.Header.Set("Content-Type", "application/json")

    httpClient := &http.Client{}
    resp, err := httpClient.Do(req)
    if err != nil {
        return &PageResult{Page: pageIdx, Err: err}
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return &PageResult{Page: pageIdx, Err: err}
    }

    var response struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
    }
    err = json.Unmarshal(body, &response)
    if err != nil {
        return &PageResult{Page: pageIdx, Err: err}
    }
    if len(response.Choices) == 0 {
        return &PageResult{Page: pageIdx, Err: errors.New("no choices in response")}
    }

    rawResponse := response.Choices[0].Message.Content
    cleanedResponse := cleanResponse(rawResponse)
    if cleanedResponse == "" {
        return &PageResult{Page: pageIdx, Err: errors.New("empty response")}
    }

    var pageResult map[string]string
    err = json.Unmarshal([]byte(cleanedResponse), &pageResult)
    if err != nil {
        return &PageResult{Page: pageIdx, Err: err}
    }

    return &PageResult{Data: pageResult}
}

func rangeInts(start, end int) []int {
    var res []int
    for i := start; i < end; i++ {
        res = append(res, i)
    }
    return res
}

func renderPdfToImages(pdfFile multipart.File) ([]image.Image, error) {
    tempFile, err := ioutil.TempFile("", "temp.pdf")
    if err != nil {
        return nil, err
    }
    defer os.Remove(tempFile.Name())

    _, err = io.Copy(tempFile, pdfFile)
    if err != nil {
        return nil, err
    }
    tempFile.Close()

    doc, err := fitz.New(tempFile.Name())
    if err != nil {
        return nil, err
    }
    defer doc.Close()

    var images []image.Image
    for n := 0; n < doc.NumPage(); n++ {
        img, err := doc.Image(n)
        if err != nil {
            return nil, err
        }
        images = append(images, img)
    }
    return images, nil
}

func processPdf(c *gin.Context) {
    file, err := c.FormFile("file")
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Please upload a PDF file"})
        return
    }
    prompt := c.PostForm("prompt")
    if prompt == "" {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Please provide a non-empty prompt"})
        return
    }

    pdfFile, err := file.Open()
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to open PDF file"})
        return
    }
    defer pdfFile.Close()

    images, err := renderPdfToImages(pdfFile)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to convert PDF to images: %v", err)})
        return
    }

    numPages := len(images)
    allResults := make(map[string]string)
    var skippedPages []int
    batchSize := 5
    model := "gemma3"
    oclient := getOpenAIClient(model)

    var wg sync.WaitGroup
    batchChan := make(chan *BatchResult, numPages/batchSize+1)

    for batchStart := 0; batchStart < numPages; batchStart += batchSize {
        batchEnd := batchStart + batchSize
        if batchEnd > numPages {
            batchEnd = numPages
        }
        batchImages := images[batchStart:batchEnd]
        var batchMessages []interface{}

        for i, img := range batchImages {
            pageIdx := batchStart + i
            b64, err := encodeImage(img)
            if err != nil {
                logger.Printf("Image processing failed for page %d: %v", pageIdx, err)
                skippedPages = append(skippedPages, pageIdx)
                continue
            }
            batchMessages = append(batchMessages, map[string]interface{}{
                "type":      "image_url",
                "image_url": map[string]string{"url": "data:image/jpeg;base64," + b64},
            })
        }

        if len(batchMessages) == 0 {
            logger.Printf("Skipping batch %d-%d: No valid images", batchStart, batchEnd-1)
            skippedPages = append(skippedPages, rangeInts(batchStart, batchEnd)...)
            continue
        }

        batchMessages = append(batchMessages, map[string]interface{}{
            "type": "text",
            "text": fmt.Sprintf("Extract plain text from these %d PDF pages. Return the results as a valid JSON object where keys are page numbers (starting from %d) and values are the extracted text for each page. Ensure the response is strictly JSON-formatted.", batchEnd-batchStart, batchStart),
        })

        wg.Add(1)
        go func(bs, be int, bm []interface{}) {
            defer wg.Done()
            result := processSingleBatch(oclient, model, bm, bs, be)
            batchChan <- result
        }(batchStart, batchEnd, batchMessages)
    }

    wg.Wait()
    close(batchChan)

    for result := range batchChan {
        if result.Err != nil {
            logger.Printf("Batch processing failed: %v", result.Err)
        }
        if result.Data != nil {
            for k, v := range result.Data {
                allResults[k] = v
            }
        }
        if len(result.SkippedPages) > 0 {
            skippedPages = append(skippedPages, result.SkippedPages...)
        }
    }

    // Retry skipped pages
    var retryWg sync.WaitGroup
    retryChan := make(chan *PageResult, len(skippedPages))
    remainingSkipped := make(map[int]struct{})
    for _, p := range skippedPages {
        remainingSkipped[p] = struct{}{}
    }

    for pageIdx := range remainingSkipped {
        retryWg.Add(1)
        go func(pi int, img image.Image) {
            defer retryWg.Done()
            result := processSinglePage(oclient, model, img, pi)
            retryChan <- result
        }(pageIdx, images[pageIdx])
    }

    retryWg.Wait()
    close(retryChan)

    var successfullyProcessed []int
    for result := range retryChan {
        if result.Err != nil {
            logger.Printf("Retry processing failed for page %d: %v", result.Page, result.Err)
            continue
        }
        if result.Data != nil {
            for k, v := range result.Data {
                allResults[k] = v
            }
            successfullyProcessed = append(successfullyProcessed, result.Page)
        }
    }

    var finalSkipped []int
    for p := range remainingSkipped {
        found := false
        for _, sp := range successfullyProcessed {
            if sp == p {
                found = true
                break
            }
        }
        if !found {
            finalSkipped = append(finalSkipped, p)
        }
    }
    sort.Ints(finalSkipped)

    if len(allResults) == 0 && len(finalSkipped) > 0 {
        c.JSON(http.StatusBadRequest, gin.H{"error": "No valid text extracted from any pages", "skipped_pages": finalSkipped})
        return
    }

    // Process with the provided prompt
    dwaniPrompt := "You are dwani, a helpful assistant. Provide a concise response in one sentence maximum. "

    resultsJSON, err := json.Marshal(allResults)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to serialize extracted text: %v", err)})
        return
    }

    combinedPrompt := fmt.Sprintf("%s\nUser prompt: %s\nExtracted text: %s", dwaniPrompt, prompt, resultsJSON)

    finalMessages := []interface{}{
        map[string]interface{}{
            "type": "text",
            "text": combinedPrompt,
        },
    }

    payload := map[string]interface{}{
        "model":      model,
        "messages": []map[string]interface{}{
            {"role": "user", "content": finalMessages},
        },
        "temperature": 0.3,
        "max_tokens":  29695,
    }
    jsonData, err := json.Marshal(payload)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to marshal final prompt: %v", err)})
        return
    }

    req, err := http.NewRequest("POST", oclient.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to create final request: %v", err)})
        return
    }
    req.Header.Set("Content-Type", "application/json")

    httpClient := &http.Client{}
    resp, err := httpClient.Do(req)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Final API request failed: %v", err)})
        return
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to read final response: %v", err)})
        return
    }

    var finalResponse struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
    }
    err = json.Unmarshal(body, &finalResponse)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to unmarshal final response: %v", err)})
        return
    }
    if len(finalResponse.Choices) == 0 {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "No choices in final response"})
        return
    }

    generatedResponse := finalResponse.Choices[0].Message.Content

    c.JSON(http.StatusOK, gin.H{
        "response":      generatedResponse,
        "extracted_text": allResults,
        "skipped_pages":  finalSkipped,
    })
}

func processMessage(c *gin.Context) {
    prompt := c.PostForm("prompt")
    if prompt == "" {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Please provide a non-empty prompt"})
        return
    }
    extractedText := c.PostForm("extracted_text")
    if extractedText == "" {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Please provide non-empty extracted text"})
        return
    }

    var allResults map[string]string
    err := json.Unmarshal([]byte(extractedText), &allResults)
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Invalid extracted text format: %v", err)})
        return
    }

    model := "gemma3"
    oclient := getOpenAIClient(model)

    dwaniPrompt := "You are dwani, a helpful assistant. Provide a concise response in one sentence maximum. "

    resultsJSON, _ := json.Marshal(allResults) // ignore err, already unmarshaled

    combinedPrompt := fmt.Sprintf("%s\nUser prompt: %s\nExtracted text: %s", dwaniPrompt, prompt, resultsJSON)

    finalMessages := []interface{}{
        map[string]interface{}{
            "type": "text",
            "text": combinedPrompt,
        },
    }

    payload := map[string]interface{}{
        "model":      model,
        "messages": []map[string]interface{}{
            {"role": "user", "content": finalMessages},
        },
        "temperature": 0.3,
        "max_tokens":  29695,
    }
    jsonData, err := json.Marshal(payload)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to marshal final prompt: %v", err)})
        return
    }

    req, err := http.NewRequest("POST", oclient.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to create final request: %v", err)})
        return
    }
    req.Header.Set("Content-Type", "application/json")

    httpClient := &http.Client{}
    resp, err := httpClient.Do(req)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Final API request failed: %v", err)})
        return
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to read final response: %v", err)})
        return
    }

    var finalResponse struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
    }
    err = json.Unmarshal(body, &finalResponse)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to unmarshal final response: %v", err)})
        return
    }
    if len(finalResponse.Choices) == 0 {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "No choices in final response"})
        return
    }

    generatedResponse := finalResponse.Choices[0].Message.Content

    c.JSON(http.StatusOK, gin.H{
        "response":      generatedResponse,
        "extracted_text": allResults,
        "skipped_pages":  []int{},
    })
}

func timingMiddleware(c *gin.Context) {
    start := time.Now()
    c.Next()
    duration := time.Since(start)
    logger.Printf("Request: %s %s took %.3f seconds", c.Request.Method, c.Request.URL.Path, duration.Seconds())
}

func main() {
    r := gin.Default()
    r.Use(timingMiddleware)

    r.POST("/process_pdf", processPdf)
    r.POST("/process_message", processMessage)

    r.Run() // listen and serve on 0.0.0.0:8080
}