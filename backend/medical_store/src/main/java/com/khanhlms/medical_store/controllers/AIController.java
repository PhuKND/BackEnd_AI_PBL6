package com.khanhlms.medical_store.controllers;

import com.khanhlms.medical_store.configuration.ApiAIConfig;
import com.khanhlms.medical_store.dtos.AI.response.PredictImageResponse;
import com.khanhlms.medical_store.dtos.response.ApiResponse;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("${app.api.prefix}/ai")
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class AIController {

    RestClient restClient;

    @PostMapping(
            value = "/image/predict",
            consumes = MediaType.MULTIPART_FORM_DATA_VALUE
    )
    public ApiResponse<PredictImageResponse> predictImage(
            @RequestParam("file") MultipartFile file
    ) {
        try {

            MultipartBodyBuilder builder = new MultipartBodyBuilder();

            builder.part("file", file.getResource())
                    .filename(file.getOriginalFilename())
                    .contentType(MediaType.parseMediaType(
                            file.getContentType() != null ?
                                    file.getContentType() :
                                    "application/octet-stream"
                    ));

            PredictImageResponse result = restClient.post()
                    .uri(ApiAIConfig.PREDICT_IMAGE_API)   // "/api/predict"
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(builder.build())
                    .retrieve()
                    .body(PredictImageResponse.class);
            System.out.println(result);

            return ApiResponse.<PredictImageResponse>builder()
                    .code(200)
                    .message("Predict Image Successfully")
                    .data(result)
                    .build();

        } catch (Exception e) {
            e.printStackTrace();
            return ApiResponse.<PredictImageResponse>builder()
                    .code(500)
                    .message("Error calling AI service: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }
}
