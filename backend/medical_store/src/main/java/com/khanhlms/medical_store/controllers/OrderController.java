package com.khanhlms.medical_store.controllers;

import com.khanhlms.medical_store.dtos.orders.request.CreateOrderRequest;
import com.khanhlms.medical_store.dtos.orders.response.CreateOrderResponse;
import com.khanhlms.medical_store.dtos.response.ApiResponse;
import com.khanhlms.medical_store.services.OrderService;
import com.khanhlms.medical_store.utills.AuthenticationUtills;
import jakarta.servlet.http.HttpServletRequest;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
@RequestMapping("${app.api.prefix}/order")
@Slf4j
public class OrderController {
    AuthenticationUtills  authenticationUtills;
    OrderService orderService;
    @PostMapping("/")
    public ApiResponse<CreateOrderResponse> createOrder(@RequestBody CreateOrderRequest order,
                                                        HttpServletRequest request) {
        String username = authenticationUtills.getUserName();

        return ApiResponse.<CreateOrderResponse>builder()
                .code(201)
                .message("Create order successfully")
                .data(this.orderService.createOrder(request, username, order))
                .build();
    }
}
