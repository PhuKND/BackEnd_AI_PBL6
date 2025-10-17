package com.khanhlms.medical_store.controllers;

import com.khanhlms.medical_store.dtos.manufacturer.requests.CreateManufaturerRequest;
import com.khanhlms.medical_store.dtos.manufacturer.response.ManufacturerResponse;
import com.khanhlms.medical_store.dtos.response.ApiResponse;
import com.khanhlms.medical_store.services.ManufacturerService;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("${app.api.prefix}")
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
@PreAuthorize("hasAuthority('ADMIN')")
public class ManufacturerController {
    ManufacturerService manufacturerService;

    @PostMapping(value = "/manufacturers", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ApiResponse<ManufacturerResponse>> handleCreateManufacturer
            (@ModelAttribute CreateManufaturerRequest  createManufaturerRequest) {
        ApiResponse<ManufacturerResponse> apiResponse = ApiResponse.<ManufacturerResponse>builder()
                .code(201)
                .message("Manufacturer created successfully")
                .data(manufacturerService.handCreateManufacturer(createManufaturerRequest))
                .build();
        return ResponseEntity.status(HttpStatus.CREATED).body(apiResponse);
    }
}
