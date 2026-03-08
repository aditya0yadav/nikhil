#ifndef MatchController_hpp
#define MatchController_hpp

#include "MatchService.hpp"
#include "oatpp/web/server/api/ApiController.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include OATPP_CODEGEN_BEGIN(ApiController)

class MatchController : public oatpp::web::server::api::ApiController {
private:
    std::shared_ptr<MatchService> m_service; 
public:
    MatchController(const std::shared_ptr<oatpp::data::mapping::ObjectMapper>& objectMapper,
                    const std::shared_ptr<MatchService>& service)
        : oatpp::web::server::api::ApiController(objectMapper), m_service(service) {}

    ENDPOINT("GET", "/scan", scanTargets) {
        auto result = m_service->runTargetMatch();
        return createResponse(Status::CODE_200, result.c_str());
    }
};

#include OATPP_CODEGEN_END(ApiController)
#endif