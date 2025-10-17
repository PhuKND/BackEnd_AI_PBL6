package com.khanhlms.medical_store.controllers;

import com.khanhlms.medical_store.dtos.cartItems.request.CreateCartItemRequest;
import com.khanhlms.medical_store.dtos.cartItems.response.CartItemResponse;
import com.khanhlms.medical_store.dtos.cartItems.response.CreateCartItemResponse;
import com.khanhlms.medical_store.dtos.response.ApiResponse;
import com.khanhlms.medical_store.services.CartItemsService;
import com.khanhlms.medical_store.utills.AuthenticationUtills;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("${app.api.prefix}")
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class CartItemsController {
    AuthenticationUtills authenticationUtills;
    CartItemsService cartItemsService;

    @ResponseStatus(HttpStatus.CREATED)
    @PostMapping("/carts/items")
    public ApiResponse<CreateCartItemResponse> creatCartItems(@RequestBody CreateCartItemRequest request){
        String username = this.authenticationUtills.getUserName();
        return ApiResponse.<CreateCartItemResponse>builder()
                .code(201)
                .message("create cart item successfully")
                .data(this.cartItemsService.handCreateCartItems(username, request))
                .build();
    }

    @GetMapping("/carts/items/mycarts")
    public ApiResponse<List<CartItemResponse>> getCartItems(
            @RequestParam(defaultValue = "0") Integer page,
            @RequestParam(defaultValue = "20") Integer size
    ) {
        String username = this.authenticationUtills.getUserName();
        Pageable pageable = PageRequest.of(page, size);
        return ApiResponse.<List<CartItemResponse>>builder()
                .data(this.cartItemsService.handMyCartItems(username,pageable ))
                .build();
    }
}
