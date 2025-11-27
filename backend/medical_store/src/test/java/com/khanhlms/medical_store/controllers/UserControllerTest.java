package com.khanhlms.medical_store.controllers;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.khanhlms.medical_store.dtos.requests.CreateUserRequest;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.model.RegisterModel;
import com.khanhlms.medical_store.services.UserService;
import com.khanhlms.medical_store.utills.ExcelUtils;
import com.khanhlms.medical_store.utills.TestResultExcelExporter;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.util.*;

import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(UserController.class)
@AutoConfigureMockMvc(addFilters = false)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@Slf4j
class UserControllerTest {

    @Autowired
    MockMvc mockMvc;

    @Autowired
    ObjectMapper objectMapper;

    @Autowired
    UserService userService; // ✅ Bean mock sẽ được inject từ TestConfig

    @TestConfiguration
    static class TestConfig {
        @Bean
        public UserService userService() {
            return mock(UserService.class);
        }
    }

    List<RegisterModel> registerCases = new ArrayList<>();
    List<Map<String, String>> results = new ArrayList<>();

    @BeforeAll
    void setup() {
        registerCases = ExcelUtils.readRegisterTestData("/data_login_test.xlsx");
    }

    @Test
    void testRegisterUserWithExcelData() throws Exception {

        for (RegisterModel testCase : registerCases) {

            CreateUserRequest request = CreateUserRequest.builder()
                    .username(testCase.getUsername())
                    .password(testCase.getPassword())
                    .repeat(testCase.getRepeat())
                    .build();

            // Tránh null, trim để so sánh chuẩn
            String expectedMsg = Optional.ofNullable(testCase.getExpectedMessage()).orElse("").trim();

            // Mock hành vi service theo từng case
            switch (testCase.getExpectedStatus()) {

                case 201 -> when(userService.handlCreateUser(any(CreateUserRequest.class)))
                        .thenReturn(true);

                case 400 -> {
                    if (expectedMsg.equalsIgnoreCase("password mismatch")) {
                        when(userService.handlCreateUser(any(CreateUserRequest.class)))
                                .thenThrow(new AppException(ErrorCode.PASSWORD_MISMATCH));
                    } else if (expectedMsg.equalsIgnoreCase("Password must be at least 8 characters")) {
                        when(userService.handlCreateUser(any(CreateUserRequest.class)))
                                .thenThrow(new AppException(ErrorCode.PASSWORD_TOO_SHORT));
                    } else {
                        when(userService.handlCreateUser(any(CreateUserRequest.class)))
                                .thenThrow(new AppException(ErrorCode.INVALID_REQUEST));
                    }
                }

                case 500 -> when(userService.handlCreateUser(any(CreateUserRequest.class)))
                        .thenThrow(new AppException(ErrorCode.USER_EXISTED));

                default -> when(userService.handlCreateUser(any(CreateUserRequest.class)))
                        .thenThrow(new AppException(ErrorCode.INVALID_REQUEST));
            }

            // Run and record result
            try {
                mockMvc.perform(post("/api/v1/users")
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(objectMapper.writeValueAsString(request)))
                        .andExpect(status().is(testCase.getExpectedStatus()))
                        .andExpect(jsonPath("$.message")
                                .value(org.hamcrest.Matchers.equalToIgnoringCase(expectedMsg)));

                addResult(testCase.getTestCaseID(), testCase.getTestCaseDescription(), "PASSED", expectedMsg);

            } catch (AssertionError e) {
                addResult(testCase.getTestCaseID(), testCase.getTestCaseDescription(), "FAILED", e.getMessage());
            }
        }
    }

    private void addResult(String testName, String desc, String status, String msg) {
        Map<String, String> record = new HashMap<>();
        record.put("Test Name", testName);
        record.put("Description", desc);
        record.put("Status", status);
        record.put("Message", msg);
        results.add(record);
    }

    @AfterAll
    void export() {
        List<String> headers = List.of("Test Name", "Description", "Status", "Message");

        TestResultExcelExporter.appendResults(
                "test-result.xlsx",
                "Users",     // hoặc "Categories", "Manufacturer", "Users"
                results,
                headers
        );
    }


}
