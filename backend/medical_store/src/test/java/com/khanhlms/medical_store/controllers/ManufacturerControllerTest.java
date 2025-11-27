package com.khanhlms.medical_store.controllers;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.khanhlms.medical_store.dtos.manufacturer.requests.CreateManufaturerRequest;
import com.khanhlms.medical_store.dtos.manufacturer.response.ManufacturerResponse;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.model.TestManufacturerModel;
import com.khanhlms.medical_store.services.ManufacturerService;
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

@WebMvcTest(ManufacturerController.class)
@AutoConfigureMockMvc(addFilters = false)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@Slf4j
class ManufacturerControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ManufacturerService manufacturerService;

    List<TestManufacturerModel> testCases = new ArrayList<>();
    List<Map<String, String>> results = new ArrayList<>();

    @TestConfiguration
    static class TestConfig {
        @Bean
        ManufacturerService manufacturerService() {
            return mock(ManufacturerService.class);
        }
    }

    @BeforeAll
    void setup() {
        testCases = ExcelUtils.readManufacturerTestData("/data_login_test.xlsx"); // sheet 2
    }

    @Test
    @WithMockUser(authorities = "ADMIN") // ✅ Bypass @PreAuthorize
    void testCreateManufacturerFromExcel() throws Exception {

        for (TestManufacturerModel testCase : testCases) {

            switch (testCase.getExpectedStatus()) {
                case 201 ->
                        when(manufacturerService.handCreateManufacturer(Mockito.any(CreateManufaturerRequest.class)))
                                .thenReturn(ManufacturerResponse.builder().name(testCase.getName()).build());

                case 400 ->
                        when(manufacturerService.handCreateManufacturer(Mockito.any(CreateManufaturerRequest.class)))
                                .thenThrow(new AppException(ErrorCode.MANUFACTURER_INVALID));

                case 401 ->
                        when(manufacturerService.handCreateManufacturer(Mockito.any(CreateManufaturerRequest.class)))
                                .thenThrow(new AppException(ErrorCode.AUTHENTICATION_EXCEPTION));

                default ->
                        when(manufacturerService.handCreateManufacturer(Mockito.any(CreateManufaturerRequest.class)))
                                .thenThrow(new AppException(ErrorCode.INVALID_REQUEST));
            }

            try {
                mockMvc.perform(multipart("/api/v1/manufacturers")  // ✅ multipart request
                                .param("name", testCase.getName()))   // ✅ gửi theo form-data
                        .andExpect(status().is(testCase.getExpectedStatus()))
                        .andExpect(jsonPath("$.message")
                                .value(org.hamcrest.Matchers.equalToIgnoringCase(testCase.getExpectedMessage().trim())));

                addResult(testCase.getTestCaseID(), testCase.getTestCaseDescription(), "PASSED", testCase.getExpectedMessage());

            } catch (AssertionError e) {
                addResult(testCase.getTestCaseID(), testCase.getTestCaseDescription(), "FAILED", e.getMessage());
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
                "Manufacturer",     // hoặc "Categories", "Manufacturer", "Users"
                results,
                headers
        );
    }



}
