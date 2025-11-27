package com.khanhlms.medical_store.controllers;

import com.khanhlms.medical_store.dtos.requests.categories.CreateCategoryRequest;
import com.khanhlms.medical_store.dtos.response.categories.CategoryResponse;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.model.CategotiesModel;
import com.khanhlms.medical_store.services.CategoriesService;
import com.khanhlms.medical_store.utills.ExcelUtils;
import com.khanhlms.medical_store.utills.TestResultExcelExporter;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.web.servlet.MockMvc;

import java.util.*;

import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(CategoriesController.class)
@AutoConfigureMockMvc(addFilters = false)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@Slf4j
class CategoriesControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private CategoriesService categoriesService;

    List<CategotiesModel> testCases = new ArrayList<>();
    List<Map<String, String>> results = new ArrayList<>();

    @TestConfiguration
    static class TestConfig {
        @Bean
        CategoriesService categoriesService() {
            return mock(CategoriesService.class);
        }
    }

    @BeforeAll
    void setup() {
        // 📌 Đọc từ Excel sheet thứ 3
        testCases = ExcelUtils.readCategoryTestData("/data_login_test.xlsx");
    }

    @Test
    @WithMockUser(authorities = "ADMIN") // Bypass security
    void testCreateCategoryFromExcel() throws Exception {

        for (CategotiesModel tc : testCases) {

            // 📌 Set behavior theo expectedStatus
            switch (tc.getExpectedStatus()) {

                case 201 ->
                        when(categoriesService.handCreteCategory(Mockito.any(CreateCategoryRequest.class)))
                                .thenReturn(CategoryResponse.builder()
                                        .name(tc.getName())
                                        .build());

                case 400 ->
                        when(categoriesService.handCreteCategory(Mockito.any(CreateCategoryRequest.class)))
                                .thenThrow(new AppException(ErrorCode.INVALID_REQUEST));

                case 401 ->
                        when(categoriesService.handCreteCategory(Mockito.any(CreateCategoryRequest.class)))
                                .thenThrow(new AppException(ErrorCode.AUTHENTICATION_EXCEPTION));

                case 500 ->
                        when(categoriesService.handCreteCategory(Mockito.any(CreateCategoryRequest.class)))
                                .thenThrow(new AppException(ErrorCode.CATEGORY_EXITSTED));

                default ->
                        when(categoriesService.handCreteCategory(Mockito.any(CreateCategoryRequest.class)))
                                .thenThrow(new AppException(ErrorCode.INVALID_REQUEST));
            }

            // 📌 Chạy Test
            try {
                mockMvc.perform(
                                multipart("/api/v1/categories")
                                        .param("name", tc.getName())
                                        .param("description", tc.getTestCaseDescription())
                        )
                        .andExpect(status().is(tc.getExpectedStatus()))
                        .andExpect(jsonPath("$.message")
                                .value(org.hamcrest.Matchers.equalToIgnoringCase(tc.getExpectedMessage().trim())));

                addResult(tc.getTestCaseID(), tc.getTestCaseDescription(), "PASSED", tc.getExpectedMessage());

            } catch (AssertionError e) {
                addResult(tc.getTestCaseID(), tc.getTestCaseDescription(), "FAILED", e.getMessage());
            }
        }
    }

    private void addResult(String id, String desc, String status, String message) {
        Map<String, String> record = new HashMap<>();
        record.put("Test Name", id);
        record.put("Description", desc);
        record.put("Status", status);
        record.put("Message", message);
        results.add(record);
    }

    @AfterAll
    void export() {
        List<String> headers = List.of("Test Name", "Description", "Status", "Message");

        TestResultExcelExporter.appendResults(
                "test-result.xlsx",
                "Categories",     // hoặc "Categories", "Manufacturer", "Users"
                results,
                headers
        );
    }


}
