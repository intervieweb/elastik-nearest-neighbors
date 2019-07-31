/*
 * Copyright [2018] [Alex Klibisz]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package org.elasticsearch.plugin.aknn;

import org.elasticsearch.action.bulk.BulkRequestBuilder;
import org.elasticsearch.action.bulk.BulkResponse;
import org.elasticsearch.action.get.GetResponse;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.node.NodeClient;
import org.elasticsearch.common.StopWatch;
import org.elasticsearch.common.inject.Inject;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.common.xcontent.XContentParser;
import org.elasticsearch.index.query.BoolQueryBuilder;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.index.query.Operator;
import org.elasticsearch.rest.BaseRestHandler;
import org.elasticsearch.rest.BytesRestResponse;
import org.elasticsearch.rest.RestController;
import org.elasticsearch.rest.RestRequest;
import org.elasticsearch.rest.RestStatus;
import org.elasticsearch.search.SearchHit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import static java.lang.Math.min;
import static org.elasticsearch.rest.RestRequest.Method.GET;
import static org.elasticsearch.rest.RestRequest.Method.POST;

public class AknnRestAction extends BaseRestHandler {

    public static String NAME = "_aknn";
    private final String NAME_SEARCH = "_aknn_search";
    private final String NAME_INDEX = "_aknn_index";
    private final String NAME_CREATE = "_aknn_create";
    private final String NAME_SEARCH_VECTOR = "_aknn_search_vector";

    // TODO: check how parameters should be defined at the plugin level.
    private final String HASHES_KEY = "_aknn_hashes";
    private final String VECTOR_KEY = "_aknn_vector";
    private final Integer K1_DEFAULT = 99;
    private final Integer K2_DEFAULT = 10;

    private String strict_mode = "STRICT";
    private String at_least_one_mode = "AT_LEAST_ONE";

    // TODO: add an option to the index endpoint handler that empties the cache.
    private Map<String, LshModel> lshModelCache = new HashMap<>();

    @Inject
    public AknnRestAction(Settings settings, RestController controller) {
        super(settings);
        controller.registerHandler(GET, "/{index}/{type}/{id}/" + NAME_SEARCH, this);
        controller.registerHandler(POST, NAME_INDEX, this);
        controller.registerHandler(POST, NAME_CREATE, this);
        controller.registerHandler(POST, "/{index}/{type}/" + NAME_SEARCH_VECTOR, this);
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest restRequest, NodeClient client) throws IOException {
        if (restRequest.path().endsWith(NAME_SEARCH))
            return handleSearchRequest(restRequest, client);
        else if (restRequest.path().endsWith(NAME_INDEX))
            return handleIndexRequest(restRequest, client);
        else if (restRequest.path().endsWith(NAME_CREATE))
            return handleCreateRequest(restRequest, client);
        else
            return handleSearchVectorRequest(restRequest, client);
    }

    public static Double euclideanDistance(List<Double> A, List<Double> B) {
        Double squaredDistance = 0.;
        for (Integer i = 0; i < A.size(); i++)
            squaredDistance += Math.pow(A.get(i) - B.get(i), 2);
        return Math.sqrt(squaredDistance);
    }

    // Cosine Distance: A B = ||A|| ||B|| cos theta

    public static Double cosineSimilarityDistance(List<Double> A, List<Double> B) {
        Double prodotto = 0.0, norma_a = 0.0, norma_b = 0.0;
        for(int i = 0; i < A.size(); i++) {
            prodotto += A.get(i) * B.get(i);
            norma_a += Math.pow(A.get(i), 2);
            norma_b += Math.pow(B.get(i), 2);
        }
        return prodotto / (Math.sqrt(norma_a) * Math.sqrt(norma_b));
    }

    public boolean validMode(String param) {
        if(param.isEmpty()) return true;
        switch(param) {
            case "STRICT":
                return true;
            case "AT_LEAST_ONE":
                return true;
            default:
                return false;
        }
    }

    private RestChannelConsumer handleSearchRequest(RestRequest restRequest, NodeClient client) throws IOException {

        StopWatch stopWatch = new StopWatch("StopWatch to Time Search Request");

        // Parse request parameters.
        stopWatch.start("Parse request parameters");
        final String index = restRequest.param("index");
        final String type = restRequest.param("type");
        final String id = restRequest.param("id");
        final Integer k1 = restRequest.paramAsInt("k1", K1_DEFAULT);
        final Integer k2 = restRequest.paramAsInt("k2", K2_DEFAULT);

        final String education_experience_title = restRequest.param("title", "");
        final String education_experience_organization = restRequest.param("organization", "");
        final String work_experience_employer = restRequest.param("employer", "");
        final String work_experience_position = restRequest.param("position", "");

        final String education_experience_title_mode = restRequest.param("title_mode", "");
        final String education_experience_organization_mode = restRequest.param("organization_mode", "");
        final String work_experience_employer_mode = restRequest.param("employer_mode", "");
        final String work_experience_position_mode = restRequest.param("position_mode", "");

        if(!validMode(education_experience_title_mode) || !validMode(education_experience_organization_mode)
        || !validMode(work_experience_employer_mode) || !validMode(work_experience_position_mode)) {
            return channel -> {
                XContentBuilder builder = channel.newBuilder();
                builder.startObject();
                builder.field("took", stopWatch.totalTime().getMillis());
                builder.field("timed_out", false);
                builder.field("error_request", "invalid query mode");
                builder.startObject("valid modes");
                builder.field("strict mode value", "STRICT");
                builder.field("at least one mode value", "AT_LEAST_ONE");
                builder.endObject();
                builder.endObject();
                channel.sendResponse(new BytesRestResponse(RestStatus.BAD_REQUEST , builder));
            };
        }

        stopWatch.stop();

        logger.info("Get query document at {}/{}/{}", index, type, id);
        stopWatch.start("Get query document");
        GetResponse queryGetResponse = client.prepareGet(index, type, id).get();
        Map<String, Object> baseSource = queryGetResponse.getSource();
        stopWatch.stop();

        logger.info("Parse query document hashes");
        stopWatch.start("Parse query document hashes");
        @SuppressWarnings("unchecked")
        Map<String, Long> queryHashes = (Map<String, Long>) baseSource.get(HASHES_KEY);
        stopWatch.stop();

        stopWatch.start("Parse query document vector");
        @SuppressWarnings("unchecked")
        List<Double> queryVector = (List<Double>) baseSource.get(VECTOR_KEY);
        stopWatch.stop();

        // Retrieve the documents with most matching hashes. https://stackoverflow.com/questions/10773581
        logger.info("Build boolean query from hashes");
        stopWatch.start("Build boolean query from hashes");
        QueryBuilder queryBuilder = QueryBuilders.boolQuery();
        for (Map.Entry<String, Long> entry : queryHashes.entrySet()) {
            String termKey = HASHES_KEY + "." + entry.getKey();
            ((BoolQueryBuilder) queryBuilder).should(QueryBuilders.termQuery(termKey, entry.getValue()));
        }

        if(!education_experience_title.isEmpty() && !education_experience_title_mode.isEmpty()) {
            logger.info("Adding title of education_experience filter in query");
            switch(education_experience_title_mode) {
                case "STRICT":
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                        "all.info.education_experience.title", education_experience_title)
                            .operator(Operator.AND)
                            .fuzzyTranspositions(false)
                            .autoGenerateSynonymsPhraseQuery(false));
                    break;
                case "AT_LEAST_ONE":
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                        "all.info.education_experience.title", education_experience_title)
                            .prefixLength(0)
                            .maxExpansions(1)
                            .fuzzyTranspositions(false)
                            .autoGenerateSynonymsPhraseQuery(false));
                    break;
                default:
                    // AT_LEAST_ONE
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                        "all.info.education_experience.title", education_experience_title)
                            .prefixLength(0)
                            .maxExpansions(1)
                            .fuzzyTranspositions(false)
                            .autoGenerateSynonymsPhraseQuery(false));
            }
        }

        if(!education_experience_organization.isEmpty() && !education_experience_organization_mode.isEmpty()) {
            logger.info("Adding organization of education_experience filter in query");
            switch(education_experience_organization_mode) {
                case "STRICT":
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                        "all.info.education_experience.organization", education_experience_organization)
                            .operator(Operator.AND)
                            .fuzzyTranspositions(false)
                            .autoGenerateSynonymsPhraseQuery(false));
                    break;
                case "AT_LEAST_ONE":
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                        "all.info.education_experience.organization", education_experience_organization)
                            .prefixLength(0)
                            .maxExpansions(1)
                            .fuzzyTranspositions(false)
                            .autoGenerateSynonymsPhraseQuery(false));
                    break;
                default:
                    // AT_LEAST_ONE
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                        "all.info.education_experience.organization", education_experience_organization)
                            .prefixLength(0)
                            .maxExpansions(1)
                            .fuzzyTranspositions(false)
                            .autoGenerateSynonymsPhraseQuery(false));
            }
        }

        if(!work_experience_employer.isEmpty() && !work_experience_employer_mode.isEmpty()) {
            logger.info("Adding employer of work_experience filter in query");
            switch(work_experience_employer_mode) {
                case "STRICT":
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                        "all.info.work_experience.employer", work_experience_employer)
                            .operator(Operator.AND)
                            .fuzzyTranspositions(false)
                            .autoGenerateSynonymsPhraseQuery(false));
                    break;
                case "AT_LEAST_ONE":
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.work_experience.employer", work_experience_employer)
                                .prefixLength(0)
                                .maxExpansions(1)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                    break;
                default:
                    // AT_LEAST_ONE
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                        "all.info.work_experience.employer", work_experience_employer)
                            .prefixLength(0)
                            .maxExpansions(1)
                            .fuzzyTranspositions(false)
                            .autoGenerateSynonymsPhraseQuery(false));
            }
        }

        if(!work_experience_position.isEmpty() && !work_experience_position_mode.isEmpty()) {
            logger.info("Adding position of work_experience filter in query");
            switch(work_experience_position_mode) {
                case "STRICT":
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                        "all.info.work_experience.position", work_experience_position)
                            .operator(Operator.AND)
                            .fuzzyTranspositions(false)
                            .autoGenerateSynonymsPhraseQuery(false));
                    break;
                case "AT_LEAST_ONE":
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                        "all.info.work_experience.position", work_experience_position)
                            .prefixLength(0)
                            .maxExpansions(1)
                            .fuzzyTranspositions(false)
                            .autoGenerateSynonymsPhraseQuery(false));
                    break;
                default:
                    // AT_LEAST_ONE
                    ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                        "all.info.work_experience.position", work_experience_position)
                            .prefixLength(0)
                            .maxExpansions(1)
                            .fuzzyTranspositions(false)
                            .autoGenerateSynonymsPhraseQuery(false));
            }
            
        }

        stopWatch.stop();

        logger.info("Execute boolean search");
        logger.info("queryBuilder is: {}", queryBuilder);
        stopWatch.start("Execute boolean search");
        SearchResponse approximateSearchResponse = client
                .prepareSearch(index)
                .setTypes(type)
                .setFetchSource("*", HASHES_KEY)
                .setQuery(queryBuilder)
                .setSize(k1)
                .get();
        stopWatch.stop();

        // Compute exact KNN on the approximate neighbors.
        // Recreate the SearchHit structure, but remove the vector and hashes.
        logger.info("Compute exact distance and construct search hits");
        stopWatch.start("Compute exact distance and construct search hits");
        List<Map<String, Object>> modifiedSortedHits = new ArrayList<>();
        for (SearchHit hit: approximateSearchResponse.getHits()) {
            Map<String, Object> hitSource = hit.getSourceAsMap();
            @SuppressWarnings("unchecked")
            List<Double> hitVector = (List<Double>) hitSource.get(VECTOR_KEY);
            hitSource.remove(VECTOR_KEY);
            hitSource.remove(HASHES_KEY);
            modifiedSortedHits.add(new HashMap<String, Object>() {{
                put("_index", hit.getIndex());
                put("_id", hit.getId());
                put("_type", hit.getType());
                put("_score", euclideanDistance(queryVector, hitVector));
                put("_score_cosine", cosineSimilarityDistance(queryVector, hitVector));
                put("_source", hitSource);
            }});
        }
        stopWatch.stop();

        logger.info("Sort search hits by exact distance");
        stopWatch.start("Sort search hits by exact distance");
        modifiedSortedHits.sort(Comparator.comparingDouble(x -> (Double) x.get("_score")));
        stopWatch.stop();

        logger.info("Timing summary\n {}", stopWatch.prettyPrint());

        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.field("timed_out", false);
            builder.startObject("hits");
            builder.field("max_score", 0);

            // In some cases there will not be enough approximate matches to return *k2* hits. For example, this could
            // be the case if the number of bits per table in the LSH model is too high, over-partioning the space.
            builder.field("total", min(k2, modifiedSortedHits.size()));
            if(!education_experience_organization_mode.isEmpty()) {
                builder.field("edu_exp_org_mode", education_experience_organization_mode);
            }
            if(!education_experience_title_mode.isEmpty()) {
                builder.field("edu_exp_title_mode", education_experience_title_mode);
            }
            if(!work_experience_employer_mode.isEmpty()) {
                builder.field("work_exp_employer_mode", work_experience_employer_mode);
            }
            if(!work_experience_position_mode.isEmpty()) {
                builder.field("work_exp_pos_mode", work_experience_position_mode);
            }
            builder.field("hits", modifiedSortedHits.subList(0, min(k2, modifiedSortedHits.size())));
            builder.endObject();
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }

    private RestChannelConsumer handleCreateRequest(RestRequest restRequest, NodeClient client) throws IOException {

        StopWatch stopWatch = new StopWatch("StopWatch to time create request");
        logger.info("Parse request");
        stopWatch.start("Parse request");

        XContentParser xContentParser = XContentHelper.createParser(
                restRequest.getXContentRegistry(), restRequest.content(), restRequest.getXContentType());
        Map<String, Object> contentMap = xContentParser.mapOrdered();
        @SuppressWarnings("unchecked")
        Map<String, Object> sourceMap = (Map<String, Object>) contentMap.get("_source");

        final String _index = (String) contentMap.get("_index");
        final String _type = (String) contentMap.get("_type");
        final String _id = (String) contentMap.get("_id");
        final String description = (String) sourceMap.get("_aknn_description");
        final Integer nbTables = (Integer) sourceMap.get("_aknn_nb_tables");
        final Integer nbBitsPerTable = (Integer) sourceMap.get("_aknn_nb_bits_per_table");
        final Integer nbDimensions = (Integer) sourceMap.get("_aknn_nb_dimensions");
        @SuppressWarnings("unchecked")
        final List<List<Double>> vectorSample = (List<List<Double>>) contentMap.get("_aknn_vector_sample");
        stopWatch.stop();

        logger.info("Fit LSH model from sample vectors");
        stopWatch.start("Fit LSH model from sample vectors");
        LshModel lshModel = new LshModel(nbTables, nbBitsPerTable, nbDimensions, description);
        lshModel.fitFromVectorSample(vectorSample);
        stopWatch.stop();

        logger.info("Serialize LSH model");
        stopWatch.start("Serialize LSH model");
        Map<String, Object> lshSerialized = lshModel.toMap();
        stopWatch.stop();

        logger.info("Index LSH model");
        stopWatch.start("Index LSH model");
        IndexResponse indexResponse = client.prepareIndex(_index, _type, _id)
                .setSource(lshSerialized)
                .get();
        stopWatch.stop();

        logger.info("Timing summary\n {}", stopWatch.prettyPrint());

        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }

    private RestChannelConsumer handleIndexRequest(RestRequest restRequest, NodeClient client) throws IOException {

        StopWatch stopWatch = new StopWatch("StopWatch to time bulk indexing request");

        logger.info("Parse request parameters");
        stopWatch.start("Parse request parameters");
        XContentParser xContentParser = XContentHelper.createParser(
                restRequest.getXContentRegistry(), restRequest.content(), restRequest.getXContentType());
        Map<String, Object> contentMap = xContentParser.mapOrdered();
        final String index = (String) contentMap.get("_index");
        final String type = (String) contentMap.get("_type");
        final String aknnURI = (String) contentMap.get("_aknn_uri");
        @SuppressWarnings("unchecked")
        final List<Map<String, Object>> docs = (List<Map<String, Object>>) contentMap.get("_aknn_docs");
        logger.info("Received {} docs for indexing", docs.size());
        stopWatch.stop();

        // TODO: check if the index exists. If not, create a mapping which does not index continuous values.
        // This is rather low priority, as I tried it via Python and it doesn't make much difference.

        // Check if the LshModel has been cached. If not, retrieve the Aknn document and use it to populate the model.
        LshModel lshModel;
        if (! lshModelCache.containsKey(aknnURI)) {

            // Get the Aknn document.
            logger.info("Get Aknn model document from {}", aknnURI);
            stopWatch.start("Get Aknn model document");
            String[] annURITokens = aknnURI.split("/");
            GetResponse aknnGetResponse = client.prepareGet(annURITokens[0], annURITokens[1], annURITokens[2]).get();
            stopWatch.stop();

            // Instantiate LSH from the source map.
            logger.info("Parse Aknn model document");
            stopWatch.start("Parse Aknn model document");
            lshModel = LshModel.fromMap(aknnGetResponse.getSourceAsMap());
            stopWatch.stop();

            // Save for later.
            lshModelCache.put(aknnURI, lshModel);

        } else {
            logger.info("Get Aknn model document from local cache");
            stopWatch.start("Get Aknn model document from local cache");
            lshModel = lshModelCache.get(aknnURI);
            stopWatch.stop();
        }

        // Prepare documents for batch indexing.
        logger.info("Hash documents for indexing");
        stopWatch.start("Hash documents for indexing");
        BulkRequestBuilder bulkIndexRequest = client.prepareBulk();
        for (Map<String, Object> doc: docs) {
            @SuppressWarnings("unchecked")
            Map<String, Object> source = (Map<String, Object>) doc.get("_source");
            @SuppressWarnings("unchecked")
            List<Double> vector = (List<Double>) source.get(VECTOR_KEY);
            source.put(HASHES_KEY, lshModel.getVectorHashes(vector));
            bulkIndexRequest.add(client
                    .prepareIndex(index, type, (String) doc.get("_id"))
                    .setSource(source));
        }
        stopWatch.stop();

        logger.info("Execute bulk indexing");
        stopWatch.start("Execute bulk indexing");
        BulkResponse bulkIndexResponse = bulkIndexRequest.get();
        stopWatch.stop();

        logger.info("Timing summary\n {}", stopWatch.prettyPrint());

        if (bulkIndexResponse.hasFailures()) {
            logger.error("Indexing failed with message: {}", bulkIndexResponse.buildFailureMessage());
            return channel -> {
                XContentBuilder builder = channel.newBuilder();
                builder.startObject();
                builder.field("took", stopWatch.totalTime().getMillis());
                builder.field("error", bulkIndexResponse.buildFailureMessage());
                builder.endObject();
                channel.sendResponse(new BytesRestResponse(RestStatus.INTERNAL_SERVER_ERROR, builder));
            };
        }

        logger.info("Indexed {} docs successfully", docs.size());
        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("size", docs.size());
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }

    private RestChannelConsumer handleSearchVectorRequest(RestRequest restRequest, NodeClient client) throws IOException {

        StopWatch stopWatch = new StopWatch("StopWatch to time search from vector request");
        logger.info("Parse request");
        stopWatch.start("Parse request");

        XContentParser xContentParser = XContentHelper.createParser(
                restRequest.getXContentRegistry(), restRequest.content(), restRequest.getXContentType());
        Map<String, Object> contentMap = xContentParser.mapOrdered();
        @SuppressWarnings("unchecked")

        final String index = restRequest.param("index");
        final String type = restRequest.param("type");
        final String aknnURI = (String) contentMap.get("_aknn_uri");
        final Integer k1 = (Integer) contentMap.get("_k1");
        final Integer k2 = (Integer) contentMap.get("_k2");

        final String education_experience_title = (String) contentMap.get("title");
        final String education_experience_organization = (String) contentMap.get("organization");
        final String work_experience_employer = (String) contentMap.get("employer");
        final String work_experience_position = (String) contentMap.get("position");

        final String education_experience_title_mode = (String) contentMap.get("title_mode");
        final String education_experience_organization_mode = (String) contentMap.get("organization_mode");
        final String work_experience_employer_mode = (String) contentMap.get("employer_mode");
        final String work_experience_position_mode = (String) contentMap.get("position_mode");

        @SuppressWarnings("unchecked")
        final List<Double> queryVector = (List<Double>) contentMap.get("_aknn_vector");
        stopWatch.stop();

        // Check if the LshModel has been cached. If not, retrieve the Aknn document and use it to populate the model.
        LshModel lshModel;
        if (! lshModelCache.containsKey(aknnURI)) {

            // Get the Aknn document.
            logger.info("Get Aknn model document from {}", aknnURI);
            stopWatch.start("Get Aknn model document");
            String[] annURITokens = aknnURI.split("/");
            GetResponse aknnGetResponse = client.prepareGet(annURITokens[0], annURITokens[1], annURITokens[2]).get();
            stopWatch.stop();

            // Instantiate LSH from the source map.
            logger.info("Parse Aknn model document");
            stopWatch.start("Parse Aknn model document");
            lshModel = LshModel.fromMap(aknnGetResponse.getSourceAsMap());
            stopWatch.stop();

            // Save for later.
            lshModelCache.put(aknnURI, lshModel);

        } else {
            logger.info("Get Aknn model document from local cache");
            stopWatch.start("Get Aknn model document from local cache");
            lshModel = lshModelCache.get(aknnURI);
            stopWatch.stop();
        }

        // Prepare documents for batch indexing.
        logger.info("Hash vector for search");
        stopWatch.start("Hash vector for search");
        
        Map<String, Long> queryHashes = lshModel.getVectorHashes(queryVector);
        
        stopWatch.stop();


        // Retrieve the documents with most matching hashes. https://stackoverflow.com/questions/10773581
        logger.info("Build boolean query from hashes");
        stopWatch.start("Build boolean query from hashes");
        QueryBuilder queryBuilder = QueryBuilders.boolQuery();
        for (Map.Entry<String, Long> entry : queryHashes.entrySet()) {
            String termKey = HASHES_KEY + "." + entry.getKey();
            ((BoolQueryBuilder) queryBuilder).should(QueryBuilders.termQuery(termKey, entry.getValue()));
        }
        stopWatch.stop();

        try {

            if(education_experience_title != null && !education_experience_title.isEmpty()
                && education_experience_title_mode != null && !education_experience_title_mode.isEmpty()) {
                logger.info("Adding title of education_experience filter in query");
                switch(education_experience_title_mode) {
                    case "STRICT":
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.education_experience.title", education_experience_title)
                                .operator(Operator.AND)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                        break;
                    case "AT_LEAST_ONE":
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.education_experience.title", education_experience_title)
                                .prefixLength(0)
                                .maxExpansions(1)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                        break;
                    default:
                        // AT_LEAST_ONE
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.education_experience.title", education_experience_title)
                                .prefixLength(0)
                                .maxExpansions(1)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                }
            }

            if(education_experience_organization != null && !education_experience_organization.isEmpty()
                && education_experience_organization_mode != null && !education_experience_organization_mode.isEmpty()) {
                logger.info("Adding organization of education_experience filter in query");
                switch(education_experience_organization_mode) {
                    case "STRICT":
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.education_experience.organization", education_experience_organization)
                                .operator(Operator.AND)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                        break;
                    case "AT_LEAST_ONE":
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.education_experience.organization", education_experience_organization)
                                .prefixLength(0)
                                .maxExpansions(1)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                        break;
                    default:
                        // AT_LEAST_ONE
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.education_experience.organization", education_experience_organization)
                                .prefixLength(0)
                                .maxExpansions(1)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                }
            }

            if(work_experience_employer != null && !work_experience_employer.isEmpty()
                && work_experience_employer_mode != null && !work_experience_employer_mode.isEmpty()) {
                logger.info("Adding employer of work_experience filter in query");
                switch(work_experience_employer_mode) {
                    case "STRICT":
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.work_experience.employer", work_experience_employer)
                                .operator(Operator.AND)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                        break;
                    case "AT_LEAST_ONE":
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                                "all.info.work_experience.employer", work_experience_employer)
                                    .prefixLength(0)
                                    .maxExpansions(1)
                                    .fuzzyTranspositions(false)
                                    .autoGenerateSynonymsPhraseQuery(false));
                        break;
                    default:
                        // AT_LEAST_ONE
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.work_experience.employer", work_experience_employer)
                                .prefixLength(0)
                                .maxExpansions(1)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                }
            }

            if(work_experience_position != null && !work_experience_position.isEmpty()
                && work_experience_position_mode != null && !work_experience_position_mode.isEmpty()) {
                logger.info("Adding position of work_experience filter in query");
                switch(work_experience_position_mode) {
                    case "STRICT":
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.work_experience.position", work_experience_position)
                                .operator(Operator.AND)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                        break;
                    case "AT_LEAST_ONE":
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.work_experience.position", work_experience_position)
                                .prefixLength(0)
                                .maxExpansions(1)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                        break;
                    default:
                        // AT_LEAST_ONE
                        ((BoolQueryBuilder) queryBuilder).must(QueryBuilders.matchQuery(
                            "all.info.work_experience.position", work_experience_position)
                                .prefixLength(0)
                                .maxExpansions(1)
                                .fuzzyTranspositions(false)
                                .autoGenerateSynonymsPhraseQuery(false));
                }
                
            }

        } catch(Exception e) {
            // Catch java.lang.NullPointerException
        }

        logger.info("Execute boolean search");
        logger.info("queryBuilder is: {}", queryBuilder);
        stopWatch.start("Execute boolean search");
        SearchResponse approximateSearchResponse = client
                .prepareSearch(index)
                .setTypes(type)
                .setFetchSource("*", HASHES_KEY)
                .setQuery(queryBuilder)
                .setSize(k1)
                .get();
        stopWatch.stop();

        // Compute exact KNN on the approximate neighbors.
        // Recreate the SearchHit structure, but remove the vector and hashes.
        logger.info("Compute exact distance and construct search hits");
        stopWatch.start("Compute exact distance and construct search hits");
        List<Map<String, Object>> modifiedSortedHits = new ArrayList<>();
        for (SearchHit hit: approximateSearchResponse.getHits()) {
            Map<String, Object> hitSource = hit.getSourceAsMap();
            @SuppressWarnings("unchecked")
            List<Double> hitVector = (List<Double>) hitSource.get(VECTOR_KEY);
            hitSource.remove(VECTOR_KEY);
            hitSource.remove(HASHES_KEY);
            modifiedSortedHits.add(new HashMap<String, Object>() {{
                put("_index", hit.getIndex());
                put("_id", hit.getId());
                put("_type", hit.getType());
                put("_score", euclideanDistance(queryVector, hitVector));
                put("_score_cosine", cosineSimilarityDistance(queryVector, hitVector));
                put("_source", hitSource);
            }});
        }
        stopWatch.stop();

        logger.info("Sort search hits by exact distance");
        stopWatch.start("Sort search hits by exact distance");
        modifiedSortedHits.sort(Comparator.comparingDouble(x -> (Double) x.get("_score")));
        stopWatch.stop();

        logger.info("Timing summary\n {}", stopWatch.prettyPrint());

        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.field("timed_out", false);
            builder.startObject("hits");
            builder.field("max_score", 0);
            try {

                if(education_experience_organization_mode != null && !education_experience_organization_mode.isEmpty()) {
                    builder.field("edu_exp_org_mode", education_experience_organization_mode);
                }

                if(education_experience_title_mode != null && !education_experience_title_mode.isEmpty()) {
                    builder.field("edu_exp_title_mode", education_experience_title_mode);
                }

                if(work_experience_employer_mode != null && !work_experience_employer_mode.isEmpty()) {
                    builder.field("work_exp_employer_mode", work_experience_employer_mode);
                }

                if(work_experience_position_mode != null && !work_experience_position_mode.isEmpty()) {
                    builder.field("work_exp_pos_mode", work_experience_position_mode);
                }

            } catch(Exception e) {
                // Catch java.lang.NullPointerException
            }
            // In some cases there will not be enough approximate matches to return *k2* hits. For example, this could
            // be the case if the number of bits per table in the LSH model is too high, over-partioning the space.
            builder.field("total", min(k2, modifiedSortedHits.size()));
            builder.field("hits", modifiedSortedHits.subList(0, min(k2, modifiedSortedHits.size())));
            builder.endObject();
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }
}
