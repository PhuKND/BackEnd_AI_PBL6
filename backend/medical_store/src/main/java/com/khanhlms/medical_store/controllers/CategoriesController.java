package com.khanhlms.medical_store.controllers;

import com.khanhlms.medical_store.dtos.requests.categories.CreateCategoryRequest;
import com.khanhlms.medical_store.dtos.response.ApiResponse;
import com.khanhlms.medical_store.dtos.response.categories.CategoryResponse;
import com.khanhlms.medical_store.services.CategoriesService;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
@RequestMapping("${app.api.prefix}")
public class CategoriesController {
    final CategoriesService categoriesService;

    @PreAuthorize("hasAuthority('ADMIN')")
    @PostMapping(value = "/categories", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ApiResponse<CategoryResponse>> createCategory
            ( @ModelAttribute CreateCategoryRequest createCategoryRequest) {
        ApiResponse<CategoryResponse> apiResponse = ApiResponse.<CategoryResponse>builder()
                .code(201)
                .message("Successfully created a new category")
                .data(this.categoriesService.handCreteCategory(createCategoryRequest))
                .build();
        return ResponseEntity.status(HttpStatus.CREATED).body(apiResponse);
    }
    @GetMapping("/categories")
    public ApiResponse<List<CategoryResponse>> getAllCategories(
            @RequestParam(defaultValue = "0") Integer page,
            @RequestParam(defaultValue = "10") Integer size
    ) {
        Pageable pageable =  PageRequest.of(page, size);
        return  ApiResponse.<List<CategoryResponse>>builder()
                .code(200)
                .message("Successfully retrieved categories")
                .data(this.categoriesService.handfindAll(pageable))
                .build();
    }
}
