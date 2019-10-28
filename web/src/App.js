import React from 'react';
import Select from 'react-select';
import axios from "axios";
import ScrollArea from "react-scrollbar"
import AttentionViewer from "./AttentionViewer.js"
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import dog_kennels from "./images/dog_kennels.png"
import flight_2 from "./images/flight_2.png"
import pets_1 from "./images/pets_1.png"
import concert_singer from "./images/concert_singer.png"
import museum_visit from "./images/museum_visit.png"
import battle_death from "./images/battle_death.png"
import student_transcripts_tracking from "./images/student_transcripts_tracking.png"
import singer from "./images/singer.png"
import cre_Doc_Template_Mgt from "./images/cre_Doc_Template_Mgt.png"
import world_1 from "./images/world_1.png"
import employee_hire_evaluation from "./images/employee_hire_evaluation.png"
import network_1 from "./images/network_1.png"
import poker_player from "./images/poker_player.png"
import real_estate_properties from "./images/real_estate_properties.png"
import course_teach from "./images/course_teach.png"
import voter_1 from "./images/voter_1.png"
import wta_1 from "./images/wta_1.png"
import orchestra from "./images/orchestra.png"
import car_1 from "./images/car_1.png"
import tvshow from "./images/tvshow.png"
import empty from "./images/empty.png"
import dev_data from "./dev.json"
import qid_checker from "./qid_checker.json"


const db_ids= [
  {label: 'dog_kennels', value: 'dog_kennels', img: dog_kennels},
  {label: 'flight_2', value: 'flight_2', img: flight_2},
  {label: 'pets_1', value: 'pets_1', img: pets_1},
  {label: 'concert_singer', value: 'concert_singer', img: concert_singer},
  {label: 'museum_visit', value: 'museum_visit', img: museum_visit},
  {label: 'battle_death', value: 'battle_death', img: battle_death},
  {label: 'student_transcripts_tracking', value: 'student_transcripts_tracking', img: student_transcripts_tracking},
  {label: 'singer', value: 'singer', img: singer},
  {label: 'cre_Doc_Template_Mgt', value: 'cre_Doc_Template_Mgt', img: cre_Doc_Template_Mgt},
  {label: 'world_1', value: 'world_1', img: world_1},
  {label: 'employee_hire_evaluation', value: 'employee_hire_evaluation', img: employee_hire_evaluation},
  {label: 'network_1', value: 'network_1', img: network_1},
  {label: 'poker_player', value: 'poker_player', img: poker_player},
  {label: 'real_estate_properties', value: 'real_estate_properties', img: real_estate_properties},
  {label: 'course_teach', value: 'course_teach', img: course_teach},
  {label: 'voter_1', value: 'voter_1', img: voter_1},
  {label: 'wta_1', value: 'wta_1', img: wta_1},
  {label: 'orchestra', value: 'orchestra', img: orchestra},
  {label: 'car_1', value: 'car_1', img: car_1},
  {label: 'tvshow', value: 'tvshow', img: tvshow}
]

class App extends React.Component {
  state = {
    db_id: '',
    nlq: '',
    result: '',
    schema_img: empty,
    attention_data: [],
    question: []
  }

  handleDBChange = (e) => {
    this.setState({
      db_id: e.value,
      schema_img: e.img
    })
  }

  handleNLQChange = (e) => {
    this.setState({
      nlq: e.target.value
    })
  }

  clicked = (e) => {
    this.setState({
      result: "  processing..."
    })
    if (this.state.db_id === '') {
      this.setState({
        result: "  plase specify db id"
      })
      return
    }

    axios.get('http://141.223.199.148:5000/service', {
      headers: { 
        'Access-Control-Allow-Origin': '*',
      },
      crossdomain: true,
      params: {db_id: this.state.db_id, question: this.state.nlq}
    }).then( response => {
      this.setState({
        result: response.data.result
      })
      if ('actions' in response.data) {
        const attention_data = []
        for (const [index, action] of response.data.actions.entries()) {
          const entry = {rule: action}
          for (const [q_index, attention] of response.data.attention[index].entries()) {
            entry[String(q_index)] = attention.toFixed(4)
          }
          attention_data.push(entry)
        }
        this.setState({
          attention_data: attention_data,
          question: response.data.question
        })
      }
    })
  }

  getContent = () => {
    const inner = [];
    dev_data.forEach((data, idx) => {
      if (data.db_id === this.state.db_id) {
        inner.push(
          <div className="ex_item">
            <div style={{textAlign: "left"}}>
              <b>SQL: </b> {data.query}
            </div>
            <div style={{textAlign: "left"}}>
              <b>NLQ: </b> {data.question}
            </div>
            <div style={{textAlign: "left"}}>
              correct rate: {qid_checker[idx]}/4
            </div>
          </div>,
        );
      }
    });
    return inner;
  }

  render(){
    return (
      <div className="App">
        <header className="App-header">
            IRNet service
        </header>
        <Select options = {db_ids} 
          placeholder='Select DB ID'
          onChange={this.handleDBChange}
        />
        <div className='schema'>
          <b> Schema:  </b>
          <img src={this.state.schema_img} alt="schema_img" />
        </div>
        <div className='examples'>
          <b> Examples:  </b>
          <ScrollArea
            speed={0.8}
            className="area"
            contentClassName="content"
            horizontal={false}
            >
          {this.getContent()}
        </ScrollArea>



        </div>
        <div className='nlq'>
          <b> NLQ: </b>
          <input className='nlq_textbox'
            onChange={this.handleNLQChange}/>
          <button onClick={this.clicked}> RUN </button>
        </div>
        <div className='result'>
          <b> Result: </b> {this.state.result}
        </div>
        <div className="attention">
          <b> Attention: </b>
          <AttentionViewer data={this.state.attention_data} question={this.state.question} />
        </div>
      </div>
    );
  }
}

export default App;
